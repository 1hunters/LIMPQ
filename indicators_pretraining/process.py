import copy
from quan.func import SwithableBatchNorm
import logging
import math
import operator
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from util import AverageMeter
from timm.utils import reduce_tensor
from quan.quantizer.lsq import LsqQuan

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def switch_bn(model, uniform_bit_width):
    for name, module in model.named_modules():
        if isinstance(module, (SwithableBatchNorm)):
            module.switch_bn(uniform_bit_width)


def set_uniform_policy_mode(model, quan_scheduler, uniform_bit_width):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            name_without_module = ".".join(name.split(".")[1:])
            if name_without_module not in quan_scheduler.excepts:
                setattr(module, 'bits', uniform_bit_width)
    switch_bn(model, uniform_bit_width)


def sample_and_activate_one_sampled_policy(model, quan_scheduler, bits_l=[]):
    config = []
    sampled_random_bit_width = max(bits_l)

    for name, module in model.named_modules():
        name_without_module = ".".join(name.split(".")[1:])

        if isinstance(module, torch.nn.Conv2d):
            if name_without_module not in quan_scheduler.excepts:
                sampled_random_bit_width = random.choice(bits_l)
                setattr(module, 'bits', sampled_random_bit_width)
                config.append(sampled_random_bit_width)
        elif isinstance(module, SwithableBatchNorm):
            if name_without_module not in quan_scheduler.excepts:
                module.switch_bn(sampled_random_bit_width)
                module.mixed_flags = True
    return config


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


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args, distill_criterion=None, alpha=1):
    target_bits = args.target_bits

    meters = [{
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'batch_time': AverageMeter()
    } for _ in range(len(target_bits) + 1)] # one for random sampling

    total_sample = len(train_loader.sampler)
    batch_size = args.dataloader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    if args.local_rank == 0:
        logger.info('Training: %d samples (%d per mini-batch)',
                    total_sample, batch_size)
    num_updates = epoch * len(train_loader)
    seed = num_updates
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        optimizer.zero_grad()
        seed = seed + 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        max_bit_outputs = None
        # previous_outputs = None
        start_time = time.time()
        for idx, bit_width in enumerate(target_bits):
            set_uniform_policy_mode(model, args.quan, bit_width)
            outputs = model(inputs)

            if distill_criterion != None: # apply inplace distillation here
                if idx == 0:
                    max_bit_outputs = outputs.detach()
                    loss = criterion(outputs, targets)
                else:
                    loss = distill_criterion(outputs, F.softmax(max_bit_outputs, dim=1))
            else:
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(meters[idx], loss, acc1, acc5, inputs.size(0), time.time() - start_time, args.world_size)

            loss.backward()
        
        sample_and_activate_one_sampled_policy(model, args.quan, target_bits)

        outputs = model(inputs)
        if distill_criterion != None:
            loss = distill_criterion(outputs, F.softmax(max_bit_outputs, dim=1))
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        
        for group in optimizer.param_groups:
            for p in group['params']:
                torch.nn.utils.clip_grad_value_(p, 1)
                if not p.grad is None and torch.sum(torch.abs(p.grad.data)) == 0.0:
                    p.grad = None
        
        optimizer.step()

        num_updates += 1

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                num_updates=num_updates, metric=meters[0]['loss'].avg)

        if args.local_rank == 0 and ((batch_idx + 1) % args.log.print_freq == 0 or batch_idx == 0):
            for m in monitors:
                for i in range(len(target_bits)):
                    p = 'Bit ' + str(target_bits[i]) + ' '
                    m.update(epoch, batch_idx + 1, steps_per_epoch, p + 'Training', {
                        p + 'Loss': meters[i]['loss'],
                        p + 'Top1': meters[i]['top1'],
                        p + 'Top5': meters[i]['top5'],
                        p + 'BatchTime': meters[i]['batch_time'],
                        p + 'LR': optimizer.param_groups[0]['lr']
                    })

            logger.info(
                "--------------------------------------------------------------------------------------------------------------")
    if args.local_rank == 0:
        for i in range(len(target_bits)):
            logger.info('==> Bit mode [%d]    Top1: %.3f    Top5: %.3f    Loss: %.3f', target_bits[i],
                        meters[i]['top1'].avg, meters[i]['top5'].avg, meters[i]['loss'].avg)
        
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    return meters[0]['top1'].avg, meters[0]['top5'].avg, meters[0]['loss'].avg


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
