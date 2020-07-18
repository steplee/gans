import torch, torch.nn as nn, torch.nn.functional as F

from gans.utils import NoiseAdderRaw

class ResnetD(nn.Module):
    def __init__(self, meta):
        super().__init__()
        from torchvision.models import resnet50

        from torchvision.ops.misc import FrozenBatchNorm2d
        norm_layer = FrozenBatchNorm2d if meta['resnetFreezeBn'] else nn.BatchNorm2d

        cs = list(resnet50(True,norm_layer=norm_layer).children())[:8]

        if meta['addNoiseToD']:
            noise = NoiseAdderRaw(.1)
            self.net = nn.Sequential(
                    noise, *cs[0:5],
                    noise, *cs[5:6],
                    noise, *cs[6:7],
                    noise, *cs[7:8],
                    noise, nn.Conv2d(2048,1,1,bias=False))
        else:
            self.net = nn.Sequential(
                    *cs,
                    nn.Conv2d(2048,1,1,bias=False))

    def forward(self, x):
        x = self.net(x)
        return x
