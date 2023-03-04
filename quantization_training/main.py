from copy import deepcopy
import logging
from pathlib import Path
import argparse
import torch
from torch.nn.modules import module
import yaml
import pickle
import quan
from quan.func import SwithableBatchNorm
import util
import os
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.models import create_model as timm_create_model
from timm.scheduler import create_scheduler
from timm.utils import distribute_bn
from torch.nn.parallel import DistributedDataParallel
from model import create_model
from process import start_cal_bn, train, validate, PerformanceScoreboard

def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir /
                           'finetune_resnet50_w3a4_12.2compression_ratio.yaml')
    
    monitors = None
    assert args.training_device == 'gpu', 'NOT SUPPORT CPU TRAINING NOW'

    if args.local_rank == 0:
        output_dir = script_dir / args.output_dir
        output_dir.mkdir(exist_ok=True)

        log_dir = util.init_logger(
            args.name, output_dir, script_dir / 'logging.conf')
        logger = logging.getLogger()

        with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
            yaml.safe_dump(args, yaml_file)

        pymonitor = util.ProgressMonitor(logger)
        tbmonitor = util.TensorBoardMonitor(logger, log_dir)
        monitors = [pymonitor, tbmonitor]

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False

    args.device = 'cuda:0'

    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    assert args.rank >= 0, 'ERROR IN RANK'
    assert args.distributed

    model = create_model(args)  # main model
    start_epoch = 0

    training_mode = getattr(args, 'training_mode', 'mixed_precision')
    assert training_mode in ['mixed_precision', 'curriculum_mixed_precision']
    
    modules_to_replace = quan.find_modules_to_quantize(model, args)
    model = quan.replace_module_by_names(model, modules_to_replace)

    if args.local_rank == 0:
        model.eval()
        tbmonitor.writer.add_graph(
            model, input_to_model=torch.randn((1, 3, 224, 224)))
        logger.info('Inserted quantizers into the original model')

    model.cuda()
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = create_optimizer(args, model)
    
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(model, args.resume.path, 'cuda', lean=args.resume.lean, optimizer=optimizer)
   
    if args.local_rank == 0:
        print(model)

    # ------------- data --------------
    train_loader, val_loader, test_loader, dist_sampler = util.data_loader.DDP_load_data(args.dataloader)
    
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    # Define loss function (criterion) and optimizer
    criterion = LabelSmoothingCrossEntropy(args.smoothing).cuda()
    
    if args.local_rank == 0:
        logger.info(('Optimizer: %s' % optimizer).replace(
            '\n', '\n' + ' ' * 11))
        # print(optimizer.get_params())
        logger.info('Total epoch: %d, Start epoch %d, Val cycle: %d', num_epochs, start_epoch, args.val_cycle)
    
    perf_scoreboard = PerformanceScoreboard(args.log.num_best_scores)
    
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)
    
    v_top1, v_top5, v_loss = 0, 0, 0
    
    if args.eval:
        validate((train_loader, val_loader), model, criterion, -1, monitors, args, batchnorm_calibration=True)
    else:  # training
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                dist_sampler.set_epoch(epoch)
            if args.local_rank == 0:
                logger.info('>>>>>>>> Epoch %3d' % epoch)
                
            t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            if args.distributed:
                torch.cuda.synchronize()
                if args.local_rank == 0:
                    logging.info("Sync running means and vars...")
                distribute_bn(model, args.world_size, True)
            
            if (epoch+1) % args.val_cycle == 0 or epoch == 0 or epoch + 15 >= num_epochs:
                v_top1, v_top5, v_loss = validate((train_loader, val_loader), model, criterion, epoch, monitors, args, batchnorm_calibration=True)
            if lr_scheduler is not None:
                lr_scheduler.step(epoch+1, v_loss)
            if args.local_rank == 0 and args.rank == 0:
                tbmonitor.writer.add_scalars(
                    'Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
                tbmonitor.writer.add_scalars(
                    'Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
                tbmonitor.writer.add_scalars(
                    'Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

                perf_scoreboard.update(v_top1, v_top5, epoch)
                is_best = perf_scoreboard.is_best(epoch)

                # save main model
                util.save_checkpoint(epoch, args.arch, model, {
                                     'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir, optimizer=optimizer)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        validate((train_loader, test_loader), model, criterion, -1, monitors, args, batchnorm_calibration=True)

    if args.local_rank == 0:
        tbmonitor.writer.close()  # close the TensorBoard
        logger.info('Program completed successfully ... exiting ...')


if __name__ == "__main__":
    main()