import torch, torch.nn as nn, torch.nn.functional as F
from gans.utils import *

#
# These are not exactly to standard.
#

#StdLayer,stdLayerD = [MinibatchStdDev], 1
StdLayer,stdLayerD = [], 0

class DcganGenerator(nn.Module):
    def __init__(self, meta):
        super().__init__()
        def block(cin,cout, k,s,p):
            return [nn.ConvTranspose2d(cin, cout, k, s, p, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(True),
                    #nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
                    #nn.BatchNorm2d(cout),
                    #nn.ReLU(True),
                    ]
        ngf=32
        self.net = nn.Sequential(
                *block(meta['latentSize'], ngf*16, 4,1,0),
                *block(ngf*16, ngf*8, 4,2,1),
                *block(ngf*8, ngf*4, 4,2,1),
                #*block(ngf*4, ngf*4, 4,2,1),
                *block(ngf*4, ngf*2, 4,2,1),
                *block(ngf*2, ngf, 4,2,1),
                nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
                #nn.Tanh()
                )
        self.net.apply(weights_init_dcgan)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

class DcganDiscriminator(nn.Module):
    def __init__(self, meta):
        super().__init__()

        def block(cin,cout, k,s,p, bn=True):
            l = [
                  nn.Conv2d(cin, cout, k, s, p, bias=False),
               *([nn.BatchNorm2d(cout)] if bn else []),
                  nn.LeakyReLU(.2,inplace=True),
                  nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
               *([nn.BatchNorm2d(cout)] if bn else []),
                  nn.LeakyReLU(.2,inplace=True)
            ]
            return l

        ndf = 32

        if meta['useMinibatchStd']:
            self.net = nn.Sequential(
                    *block(3, ndf, 5,2,2, bn=False), # 64
                    *block(ndf, ndf*2, 3,2,1), # 32
                    *block(ndf*2, ndf*8, 3,2,1), # 16
                    *StdLayer,
                    *block(ndf*8+stdLayerD, ndf*16, 3,2,1,bn=False), # 8
                    *StdLayer,
                    *block(ndf*16+stdLayerD, ndf*16, 3,2,1), # 4
                    nn.Conv2d(ndf*16, 1, 1, bias=False))
        else:
            self.net = nn.Sequential(
                    *block(3, ndf, 5,2,2, bn=False), # 64
                    *block(ndf, ndf*2, 3,2,1), # 32
                    *block(ndf*2, ndf*8, 3,2,1), # 16
                    *block(ndf*8, ndf*16, 3,2,1,bn=False), # 8
                    *block(ndf*16, ndf*16, 3,2,1), # 4
                    nn.Conv2d(ndf*16, 1, 1, bias=False))

        self.net.apply(weights_init_dcgan)
        #self.net.apply(weights_init_ortho)

    def forward(self, x):
        return self.net(x)
