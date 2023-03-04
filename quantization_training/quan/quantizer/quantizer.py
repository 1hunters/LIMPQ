import torch


class Quantizer(torch.nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, bit_width, *args, **kwargs):
        pass

    def forward(self, x, bits):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x, bits):
        return x
    
    def init_from(self, x, bit_width, *args, **kwargs):
        return
