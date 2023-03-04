import torch

from .quantizer import Quantizer


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


def quant_operator(x, s, thd_neg, thd_pos, floor):
    s_grad_scale = 1.0 / ((thd_pos * x.numel()) ** 0.5)
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
        self.s = torch.nn.Parameter(torch.ones(1))

    def init_from(self, x, bit_width, *args, **kwargs):
        self.bit_width = bit_width
        self.register_buffer('init_state', torch.zeros(1))

        if x is not None:
            if self.per_channel:
                self.s = torch.nn.Parameter(
                    x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            else:
                self.s = torch.nn.Parameter(torch.ones(1))
                
                mean = x.detach().mean()
                std = x.detach().std()
                s_init = torch.max((mean-3*std).abs(), (mean+3*std).abs())/2**(bit_width-1)
                self.s.data.copy_(s_init)
                self.init_state.fill_(1)
        else:
            self.s = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, bits):
        if bits is None:
            return x
        
        thd_neg, thd_pos = compute_thd(self, bits)

        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            self.s.data.copy_(x.detach().abs().mean() * 2 / (thd_pos ** 0.5))
        
        x = quant_operator(x, self.s, thd_neg, thd_pos, False)
        return x