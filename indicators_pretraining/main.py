import logging
from pathlib import Path
import torch
import yaml
# import meta_training_quan
import quan
# from meta_training_quan.func import SwithableBatchNorm
import util
import os
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import distribute_bn
from torch.nn.parallel import DistributedDataParallel as DDP
from model import create_model
from process import train, PerformanceScoreboard

def init_activation_scale_factors(model):
    pass


def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir /
                           'config_resnet50.yaml')
    
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
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    assert args.rank >= 0, 'ERROR IN RANK'
    assert args.distributed

    model = create_model(args)  # main model
    start_epoch = 0

    modules_to_replace = quan.find_modules_to_quantize(model, args)
    model = quan.replace_module_by_names(model, modules_to_replace)

    if args.local_rank == 0:
        tbmonitor.writer.add_graph(
            model, input_to_model=torch.randn((1, 3, 224, 224)))
        logger.info('Inserted quantizers into the original model')

    model.cuda()
    model = DDP(model, device_ids=[
                args.local_rank], find_unused_parameters=True)

    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, 'cuda', lean=args.resume.lean)

   
    if args.local_rank == 0:
        print(model)
    
    if args.freeze_weights:
        for name, para in model.named_parameters():
            if 'quan_' not in name and 'bn' not in name:
                para.requires_grad = False

    # ------------- data --------------
    train_loader, val_loader, test_loader, dist_sampler = util.data_loader.DDP_load_data(
        args.dataloader)

    # Define loss function (criterion) and optimizer
    criterion = LabelSmoothingCrossEntropy(args.smoothing).cuda()
    distill_criterion = SoftTargetCrossEntropy().cuda()
    distill_criterion = None

    optimizer = create_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    if args.local_rank == 0:
        logger.info(('Optimizer: %s' % optimizer).replace(
            '\n', '\n' + ' ' * 11))
        # print(optimizer.get_params())
        logger.info('Total epoch: %d, Start epoch %d, Val cycle: %d', num_epochs, start_epoch, args.val_cycle)
    perf_scoreboard = PerformanceScoreboard(args.log.num_best_scores)
    
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)
    
    for epoch in range(start_epoch, num_epochs):
        if args.distributed:
            dist_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            
        t_top1, t_top5, t_loss = train(train_loader, model, criterion, optimizer,
                                                lr_scheduler, epoch, monitors, args, distill_criterion=distill_criterion)
        if args.distributed:
            torch.cuda.synchronize()
            if args.local_rank == 0:
                logging.info("Sync running means and vars...")
            distribute_bn(model, args.world_size, True)

        if lr_scheduler is not None:
            lr_scheduler.step(epoch+1, v_loss)
        if args.local_rank == 0:
            tbmonitor.writer.add_scalars(
                'Train_vs_Validation/Loss', {'train': t_loss}, epoch)
            tbmonitor.writer.add_scalars(
                'Train_vs_Validation/Top1', {'train': t_top1}, epoch)
            tbmonitor.writer.add_scalars(
                'Train_vs_Validation/Top5', {'train': t_top5}, epoch)

            perf_scoreboard.update(t_top1, t_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)

            # save main model
            util.save_checkpoint(epoch, args.arch, model, {
                                    'top1': t_top1, 'top5': t_top5}, is_best, args.name, log_dir)


if __name__ == "__main__":
    main()