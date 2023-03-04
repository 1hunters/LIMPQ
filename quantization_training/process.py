import copy

from torch.nn.modules.batchnorm import BatchNorm2d
from quan.func import SwithableBatchNorm
import logging
import math
import operator
import time
import torch
import torch.nn as nn
import numpy as np
import random
from util import AverageMeter
from timm.utils import reduce_tensor
from quan.quantizer.lsq import LsqQuan, quant_operator, compute_thd, round_pass
from quan.func import QuanConv2d, IdentityQuan, LsqQuan

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_meter(meter, loss, acc1, acc5, size, batch_time, world_size):
    reduced_loss = reduce_tensor(loss.data, world_size)
    reduced_top1 = reduce_tensor(acc1, world_size)
    reduced_top5 = reduce_tensor(acc5, world_size)
    meter['loss'].update(reduced_loss.item(), size)
    meter['top1'].update(reduced_top1.item(), size)
    meter['top5'].update(reduced_top5.item(), size)
    meter['batch_time'].update(batch_time)


def start_cal_bn(model, train_loader, args, cal_limit=100):
    model.eval()
    for n, m in model.named_modules():
        if isinstance(m, BatchNorm2d):

            m.reset_running_stats()
            m.training = True
            m.momentum = None
    n = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if n > cal_limit:
            break

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        model(inputs)

        if args.local_rank == 0:
            print("bn cal...", batch_idx)
        n += 1

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args, train_step_size_only=False):

    meter = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    } 

    total_sample = len(train_loader.sampler)
    batch_size = args.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    if args.local_rank == 0:
        logger.info('Training: %d samples (%d per mini-batch)',
                    total_sample, batch_size)
    num_updates = epoch * len(train_loader)
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()

        start_time = time.time()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        
        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        update_meter(meter, loss, acc1, acc5, inputs.size(0), time.time() - start_time, args.world_size)

        loss.backward()
        
        optimizer.step()
        num_updates += 1

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                num_updates=num_updates, metric=meter['loss'].avg)

        if args.local_rank == 0 and (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:

                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': meter['loss'],
                    'Top1': meter['top1'],
                    'Top5': meter['top5'],
                    'BatchTime': meter['batch_time'],
                    'LR': optimizer.param_groups[0]['lr']
                })

            logger.info(
                "--------------------------------------------------------------------------------------------------------------")
    if args.local_rank == 0:
        logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f', meter['top1'].avg, meter['top5'].avg, meter['loss'].avg)
    
    return meter['top1'].avg, meter['top5'].avg, meter['loss'].avg


def validate(loaders, model, criterion, epoch, monitors, args, nr_random_sample=0, alpha=1, batchnorm_calibration=True):
    train_loader, data_loader = loaders

    if batchnorm_calibration:
        calibration_samples = 6000
        start_cal_bn(model, train_loader=train_loader, args=args, cal_limit=calibration_samples//train_loader.batch_size)
        
    meter = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    }

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    if args.local_rank == 0:
        logger.info('Validation: %d samples (%d per mini-batch)',
                    total_sample, batch_size)

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            start_time = time.time()
            outputs = model(inputs)
            loss = alpha * criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meter, loss, acc1, acc5, inputs.size(0), time.time() - start_time, args.world_size)

            if args.local_rank == 0:
                if (batch_idx + 1) % args.log.print_freq == 0:
                    for m in monitors:
                        m.update(epoch, batch_idx + 1, steps_per_epoch, 'Val', {
                            'Loss': meter['loss'],
                            'Top1': meter['top1'],
                            'Top5': meter['top5'],
                            'BatchTime': meter['batch_time'],
                        })
                    logger.info(
                        "--------------------------------------------------------------------------------------------------------------")
    if args.local_rank == 0:
        logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f', meter['top1'].avg, meter['top5'].avg, meter['loss'].avg)
        
    return meter['top1'].avg, meter['top5'].avg, meter['loss'].avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
