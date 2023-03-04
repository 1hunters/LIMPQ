import logging

from .func import *
from .quantizer import *
import torch
import torch.nn as nn

def quantizer(default_cfg, this_cfg=None, excepts_bits_width=None):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v
    
    assert target_cfg['mode'] == 'lsq'
    q = LsqQuan

#     if target_cfg['bit'] is None:
# #         if excepts_bits_width is not None:
# #             target_cfg['bit'] = excepts_bits_width
# #         else:
#         q = IdentityQuan

    target_cfg.pop('mode')
    return q(**target_cfg)


def find_modules_to_quantize(model, args):
    replaced_modules = dict()
    weight_bits = args.weight_bits
    act_bits = args.act_bits

    idx = 0
    for name, module in model.named_modules():

        if type(module) in QuanModuleMapping.keys():
            key = type(module) 
            if name in args.quan.excepts:
                if isinstance(module, nn.Conv2d):
                    bits = (8, 32)
                else:
                    bits = (8, 8)
                replaced_modules[name] = QuanModuleMapping[key](
                    module,
                    bits=bits,
                    quan_w_fn=quantizer(args.quan.weight,
                                        args.quan.excepts[name].weight, 8),
                    quan_a_fn=quantizer(args.quan.act,
                                        args.quan.excepts[name].act, 8),
                )
            else:

                replaced_modules[name] = QuanModuleMapping[key](
                    module,
                    bits=(weight_bits[idx], act_bits[idx]),
                    quan_w_fn=quantizer(args.quan.weight),
                    quan_a_fn=quantizer(args.quan.act),
                )

                logging.info("Layer %s is quantized to [%d]W[%d]A", name, weight_bits[idx], act_bits[idx])
                idx += 1
        elif name in args.quan.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)
    
    assert idx == len(weight_bits)
    return replaced_modules

def replace_module_by_names(model, modules_to_replace):
    def helper(child: torch.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model