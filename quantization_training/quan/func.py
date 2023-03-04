import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import *


class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, bits=None, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode
                         )
        self.wbits, self.abits = bits

        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn if self.abits < 32 else IdentityQuan()
        self.bits = None
        self.weight = torch.nn.Parameter(m.weight.detach())
        
        assert self.wbits > 1 and self.abits > 1

        self.quan_w_fn.init_from(m.weight, self.wbits)
        self.quan_a_fn.init_from(None, self.abits)

        self.weight_norm = False
        self.act_rescale = False

        if self.act_rescale:
            self.act_rescale_p = nn.Parameter(torch.ones(1))
        
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        weight = self.weight

        if self.weight_norm:
            w_mean = self.weight.data.mean().cuda()
            w_std = self.weight.data.std().cuda()
            weight = self.weight.add(-w_mean).div(w_std)

        quantized_weight = self.quan_w_fn(weight, self.wbits)
        quantized_act = self.quan_a_fn(x, self.abits)

        return F.conv2d(input=quantized_act, weight=quantized_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, bits=None, quan_w_fn=None, quan_a_fn=None):
        assert isinstance(m, torch.nn.Linear)
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.bits = None
        self.weight = torch.nn.Parameter(m.weight.detach())
        self.wbits, self.abits = bits
        assert self.wbits is not None

        self.quan_w_fn.init_from(m.weight, self.wbits)
        self.quan_a_fn.init_from(None, self.abits)
        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight, self.wbits)
        quantized_act = self.quan_a_fn(x, self.abits)

        weight_scale = 1.0 / (self.out_features) ** 0.5
        weight_scale /= torch.std(quantized_weight.detach())

        bias = self.bias

        if self.training: # from SAT
            quantized_weight.mul_(weight_scale)
        if bias is not None and not self.training:
            bias = bias / weight_scale
        
        return F.linear(quantized_act, quantized_weight, bias)
    

class SwithableBatchNorm():
    pass
QuanModuleMapping = {
    torch.nn.Conv2d: QuanConv2d,
    torch.nn.Linear: QuanLinear,
    timm.models.layers.linear.Linear: QuanLinear,
}