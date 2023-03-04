import torch
import copy
import torch.nn.functional as F

class QuanConv2d(torch.nn.Conv2d):
    def __init__(self, m: torch.nn.Conv2d, bits_list=[], quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode
                         )
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.bits = None
        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight, bits_list)
        self.quan_a_fn.init_from(None, bits_list)

        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight, self.bits, is_activation=False)
        quantized_act = self.quan_a_fn(x, self.bits, is_activation=True)
        
        return F.conv2d(input=quantized_act, weight=quantized_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class QuanLinear(torch.nn.Linear):
    def __init__(self, m: torch.nn.Linear, bits_list=[], quan_w_fn=None, quan_a_fn=None):
        assert type(m) == torch.nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.bits = None
        self.weight = torch.nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from(m.weight, bits_list)
        self.quan_a_fn.init_from(None, bits_list)

        if m.bias is not None:
            self.bias = torch.nn.Parameter(m.bias.detach())
        

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight, self.bits, is_activation=False)
        quantized_act = self.quan_a_fn(x, self.bits, is_activation=True)
        return F.linear(quantized_act, quantized_weight, self.bias)


class SwithableBatchNorm(torch.nn.Module):
    def __init__(self, m : torch.nn.BatchNorm2d, num_features, bits_list=None):
        super(SwithableBatchNorm, self).__init__()
        self.num_features = num_features
        self.bits_list = bits_list
        self.bit_width = -1
        self.mixed_flags = False
        
        if bits_list is not None:
            bn_list = []
            for _ in range(len(bits_list)):
                bn = copy.deepcopy(m)
                bn_list.append(bn)
            
            self.bns = torch.nn.ModuleList(bn_list)
        else:
            self.bn = copy.deepcopy(m)
    
    def switch_bn(self, bit_width):
        if self.bits_list is not None:
            self.bit_width = bit_width

    def forward(self, input):
        if self.bits_list is not None:
            if self.bit_width not in self.bits_list:
                idx = 0
            else:
                idx = self.bits_list.index(self.bit_width)
            
            o = self.bns[idx](input)

            return o
        
        return self.bn(input)

QuanModuleMapping = {
    torch.nn.Conv2d: QuanConv2d,
    torch.nn.Linear: QuanLinear,
    torch.nn.BatchNorm2d: SwithableBatchNorm
}