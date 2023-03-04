import torch.nn as nn
import torch
from collections import OrderedDict

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            dw_conv = nn.Sequential(
                OrderedDict([
                ('conv', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)), 
                ('bn', nn.BatchNorm2d(inp)),
                ('relu', nn.ReLU(inplace=True))
            ]))

            pw_conv = nn.Sequential(
                OrderedDict([
                ('conv', nn.Conv2d(inp, oup, 1, 1, 0, bias=False)), 
                ('bn', nn.BatchNorm2d(oup)),
                ('relu', nn.ReLU(inplace=True))
            ]))
            
            return nn.Sequential(OrderedDict(
                [('dw_conv', dw_conv),
                 ('pw_conv', pw_conv),
                ]
            ))
        
        self.first_conv = conv_bn(3, 32, 2)

        channels = [64, 128, 256, 512, 1024]
        depths = [1, 2, 2, 6, 2]
        strides = [1, 2, 2, 2, 2]

        in_channel = 32

        features = []

        for stage_id, (depth, channel, stride) in enumerate(zip(depths, channels, strides)):
            ops = []
            first_layer = conv_dw(inp=in_channel, oup=channel, stride=stride)
            ops.append(('unit1', first_layer))

            print(in_channel, channel)

            in_channel = channel

            for layer_id in range(1, depth):
                ops.append((f'unit{layer_id+1}', conv_dw(inp=in_channel, oup=channel, stride=stride)))
                print(in_channel, channel)
            
            features.append((f'stage{stage_id+1}', nn.Sequential(OrderedDict(ops))))
        
        self.features = nn.Sequential(OrderedDict(features))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_v1(pretrained=True):
    model = MobileNetV1()

    if pretrained:
        state_dict = torch.load("/home/ctang/full_precision_models/mobilenetv1_71.8.pth.tar")
        import collections
        new_state_dict = collections.OrderedDict()
        state_dict = state_dict['state_dict']
        for key, value in state_dict.items():
            new_state_dict[key[7:]] = value
        model.load_state_dict(new_state_dict)
    
    return model