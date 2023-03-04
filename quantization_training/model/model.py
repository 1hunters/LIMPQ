import logging

from .mobilenet import *
import timm

def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.dataloader.dataset == 'imagenet':
        if args.arch == 'resnet18':
            model = timm.create_model('gluon_resnet18_v1b', pretrained=args.pre_trained)
        elif args.arch == 'resnet50':
            model = timm.create_model('gluon_resnet50_v1b', pretrained=args.pre_trained)
        elif args.arch == 'mobilenetv1':
            model = mobilenet_v1(pretrained=args.pre_trained)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    logger.info(msg)

    return model