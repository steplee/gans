import torch, torch.nn as nn, torch.nn.functional as F
from gans.utils import *

#ReLU = lambda i:  nn.ReLU(i)
ReLU =lambda i:  nn.LeakyReLU(.2,i)

class StyleBlock(nn.Module):
    def __init__(self, meta, cIn, cOut, firstConv=True, iniNoiseScale=.1):
        super().__init__()

        if firstConv:
            self.conv1 = nn.Conv2d(cIn, cIn, 3, padding=1)
        else:
            self.conv1 = nn.Sequential()

        self.conv2 = nn.Conv2d(cIn, cOut, 3, padding=1)
        self.noise1 = NoiseAdder(cIn,  iniNoiseScale)
        self.noise2 = NoiseAdder(cOut, iniNoiseScale)
        self.xform_w1 = nn.Linear(meta['latentSize'], cIn*2,  bias=True)
        self.xform_w2 = nn.Linear(meta['latentSize'], cOut*2, bias=True)
        self.relu = ReLU(True)
        self.cIn, self.cOut = cIn, cOut

    def forward(self, x, ww):
        #print(self.scale_noise1)
        b,c,h,w = x.size()

        x = self.conv1(x)
        x = self.noise1(x)
        mu1,std1 = self.xform_w1(ww).split(self.cIn, 1)
        x = self.relu(x)
        stats1 = self.xform_w1(ww)
        x = adain_with_stats(x, mu1, std1+.7)

        x = self.conv2(x)
        #x = self.pn(x)
        x = self.noise2(x)
        x = self.relu(x)
        mu2,std2 = self.xform_w2(ww).split(self.cOut, 1)
        x = adain_with_stats(x, mu2, std2+.7)

        return x

class StyleGanGenerator(nn.Module):
    def __init__(self, meta):
        super().__init__()

        self.const = nn.Parameter(torch.randn(1, 512, 4, 4)/30)

        mapLatent = []
        for i in range(meta['numLatentMaps']):
            mapLatent.append(nn.Linear(meta['latentSize'], meta['latentSize'], bias=False))
            if i < meta['numLatentMaps']-1:
                if meta['styleganLatentBatchNorm']: mapLatent.append(nn.BatchNorm1d(meta['latentSize']))
                mapLatent.append(ReLU(True))
        self.mapLatent = nn.Sequential(*mapLatent)
        #self.mapLatent.apply(latent_init)

        self.upsample = Upsample2d()
        blocks = [
            StyleBlock(meta, 512, 512, firstConv=False, iniNoiseScale=.1), # 4
            StyleBlock(meta, 512, 256, iniNoiseScale=.01),                  # 8
            StyleBlock(meta, 256, 256, iniNoiseScale=.01),                  # 16
            StyleBlock(meta, 256, 128, iniNoiseScale=.005),                 # 32
            StyleBlock(meta, 128, 128, iniNoiseScale=.001),                 # 64
        ]
        if meta['resolution'] >= 256:
            blocks.append(StyleBlock(meta, 128, 128,  iniNoiseScale=.001))  # 128
            blocks.append(StyleBlock(meta, 128, 64,  iniNoiseScale=.001))   # 256
        else:
            blocks.append(StyleBlock(meta, 128, 64, iniNoiseScale=.001))    # 128
        if meta['resolution'] >= 512:
            blocks.append(StyleBlock(meta, 128, 64,  iniNoiseScale=.001))   # 512
        self.blocks = nn.ModuleList(blocks)

        self.finale = nn.Sequential(
                nn.Conv2d(64, 3, 1, bias=False),
                #nn.Tanh()
            )

    def forward(self, z):
        w = self.mapLatent(z)
        x = self.const.repeat(z.size(0), 1, 1, 1)
        for i,blk in enumerate(self.blocks):
            if i > 0:
                x = self.upsample(x)
            x = blk(x, w)

        x = self.finale(x)
        return x
