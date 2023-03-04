import torch
import torch.distributed as dist
from .quantizer import Quantizer

# code from https://github.com/zhutmost/lsq-net

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


def quant_operator(x, s, thd_neg, thd_pos, scale_grad=False):
    if scale_grad:
        s_grad_scale = 1.0 / ((thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(s, s_grad_scale)
    else:
        s_scale = s

    x = x / s_scale
    x = x.clamp(min=thd_neg, max=thd_pos)
    x = round_pass(x)
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
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, scale_grad=True):
        super().__init__(bit)
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1))
        self.bit_list = (8, )
        self.bits = 2
        self.scale_gradient = scale_grad

    def init_from(self, x, bit_list, *args, **kwargs):
        self.bit_list = tuple(bit_list)
        self.register_buffer('init_state', torch.zeros(len(bit_list)))

        if x is not None:
            if self.per_channel:
                self.s = torch.nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            else:
                self.s = torch.nn.Parameter(torch.ones(len(bit_list)))
                for i in range(len(self.bit_list)):
                    mean = x.detach().mean()
                    std = x.detach().std()
                    s_init = torch.max((mean-3*std).abs(), (mean+3*std).abs())/2**(bit_list[i]-1)
                    self.s[i].data.copy_(s_init)
                    self.init_state[i].fill_(1)
        else:
            self.s = torch.nn.Parameter(torch.ones(len(bit_list)))

    def __repr__(self):
        return f'LSQ quantizer. Bit-width candidates: {self.bit_list}, all positive: {self.all_positive}, symmetric: {self.symmetric}'

    def forward(self, x, bits):
        if bits is None or bits >= 32:
            return x

        idx = self.bit_list.index(bits)
        thd_neg, thd_pos = compute_thd(self, bits)
        self.bits = bits

        if self.training and self.init_state[idx] == 0:
            self.init_state[idx].fill_(1)
            init_val = x.detach().abs().mean() * 2 / (thd_pos ** 0.5)
            dist.all_reduce(init_val)
            init_val /= dist.get_world_size()
            self.s[idx].data.copy_(init_val)

        s = self.s[idx] # switch to s[bits]
        s = torch.clamp(s, min=1e-5, max=1.)
        x = quant_operator(x, s, thd_neg, thd_pos, scale_grad=self.scale_gradient)

        return x