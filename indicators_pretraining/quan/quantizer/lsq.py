import torch
import torch.nn as nn
from .quantizer import Quantizer


def gaussian_initialization(x, b, kernel_wise):
    if kernel_wise:
        mean = x.detach().mean(dim=list(range(1, x.dim())), keepdim=True)
        std = x.detach().std(dim=list(range(1, x.dim())), keepdim=True)
    else:
        mean = x.detach().mean()
        std = x.detach().std()
    s_init = torch.max((mean-3*std).abs(), (mean+3*std).abs())/2**(b-1)
    return s_init


def scale_init():
    pass


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y-y_grad).detach()+y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y-y_grad).detach()+y_grad
    

def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y-y_grad).detach()+y_grad


def quant_operator(x, s, thd_neg, thd_pos, s_grad_scale):
    s_scale = grad_scale(s, s_grad_scale)
    x = x / s_scale
    x = torch.clamp(x, thd_neg, thd_pos)
    x = round_pass(x) # get max_bit weight
    x = x * s_scale
    return x


def compute_thd(self, bits):
    if self.all_positive:
            assert not self.symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            thd_neg = 0
            thd_pos = 2 ** bits - 1
    else:
        if self.symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            thd_neg = - 2 ** (bits - 1) + 1
            thd_pos = 2 ** (bits - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            thd_neg = - 2 ** (bits - 1)
            thd_pos = 2 ** (bits - 1) - 1
    return thd_neg, thd_pos


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, floor_weight=True):
        super().__init__(bit)
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x: torch.Tensor, bit_list: list, *args, **kwargs):
        self.bit_list = bit_list
        _n = len(bit_list)
        _t = torch.ones(_n)
        self.register_buffer('init_state', torch.zeros(_n))
        if not self.per_channel or x is None:
            s_grad_scale= torch.zeros(_n)

        # only kernel-wise for weights
        if x is not None:
            if self.per_channel:
                _t = [x.shape[i] if i == 0 else 1 for i in range(x.dim())]
                _t.insert(0, _n)
                _t = torch.zeros(_t)
                s_grad_scale = torch.zeros_like(_t)

            for i in range(_n):
                self.init_state[i].fill_(1)
                s_init = gaussian_initialization(x=x, b=bit_list[i], kernel_wise=self.per_channel)
                _t[i] = s_init
                _, thd_pos = compute_thd(self, bit_list[i])
                if self.per_channel:
                    for c in range(s_grad_scale[i].shape[0]):
                        s_grad_scale[i][c, ] = 1.0 / ((thd_pos * x[c].numel()) ** 0.5)
                else:
                    s_grad_scale[i] = 1.0 / ((thd_pos * x.numel()) ** 0.5)
        
        self.s = nn.Parameter(_t)
        self.register_buffer('s_grad_scale', s_grad_scale)

    def forward(self, x, bits, is_activation=False):
        if bits is None:
            return x
        
        idx = self.bit_list.index(bits)
        thd_neg, thd_pos = compute_thd(self, bits)

        if self.training and self.init_state[idx] == 0:
            self.init_state[idx].fill_(1)
            
            s_init = x.detach().abs().mean() * 2 / (thd_pos ** 0.5)
            self.s[idx].data.copy_(s_init)
            self.s_grad_scale[idx].fill_(1.0 / ((thd_pos * x.numel()) ** 0.5))

        s = self.s[idx] # switch to s[bits]
        grad_scale = self.s_grad_scale[idx]

        s = torch.clamp(s, 1e5, 1.)

        x = quant_operator(x, s, thd_neg, thd_pos, grad_scale)
        return x