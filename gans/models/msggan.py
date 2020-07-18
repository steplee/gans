import torch, torch.nn as nn, torch.nn.functional as F
from gans.base import GanBase
from gans.utils import *

ReLU = lambda: nn.LeakyReLU(.2,inplace=True)
#ReLU = lambda: nn.SELU(inplace=True)
#useBn = False

init_fn = weights_init_dcgan
#init_fn = weights_init_he
#init_fn = lambda m: m

StdLayer,stdLayerD = lambda: [MinibatchStdDev()], 1
#StdLayer,stdLayerD = lambda: [], 0

def safe_norm(x, dim, eps=1e-7):
    return ((x**2).sum(dim=dim) + eps).sqrt()

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockG(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, up=None, upConv=None, groups=1,
                 base_width=64, dilation=1, bn=None, doStd=True):
        super().__init__()
        if bn is None: bn = lambda x: nn.Sequential()
        if groups != 1 or base_width != 64: raise ValueError('BasicBlockD only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlockD")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn(planes)
        self.relu = ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn(planes)
        self.up = up
        #self.upConv = upConv
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        '''
        if self.upConv is not None:
            identity = self.upConv(identity)
        out += identity
        '''
        if self.up:
            out = self.up(out)
        out = self.relu(out)
        return out

class MsgFirstG(nn.Module):
    def __init__(self, meta):
        super().__init__()

        useBn = True
        bn = nn.BatchNorm2d if useBn else lambda n: nn.Sequential()
        #bn = lambda c: nn.InstanceNorm2d(c, affine=False, track_running_stats=False)
        #bn = lambda c: nn.InstanceNorm2d(c, affine=True, track_running_stats=False)
        #bn = lambda c: nn.LocalResponseNorm(c)

        upsample = Upsample2d()
        #upsample = Upsample2d(mode='bilinear')

        def make_up_conv(cin, cout):
            return conv1x1(cin, cout)
        def finale(cin, bn=useBn):
            return nn.Conv2d(cin,3, 1,1,0, bias=False)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(meta['latentSize'], 512, 4, 4, 0, bias=False),
            *([nn.BatchNorm2d(512)] if bn else []),
            ReLU()
        )

        self.blocks = nn.ModuleList([
                BasicBlockG(512, 512, bn=bn, up=upsample, upConv=make_up_conv(512,512)), # 4 > 8
                BasicBlockG(512, 512, bn=bn, up=upsample, upConv=make_up_conv(512,512)), # 8 > 16
                BasicBlockG(512, 256, bn=bn, up=upsample, upConv=make_up_conv(512,256)), # 16 > 32
                BasicBlockG(256, 128, bn=bn, up=upsample, upConv=make_up_conv(256,128)), # 32 > 64
                BasicBlockG(128, 64,  bn=bn, up=upsample, upConv=make_up_conv(128,64)),  # 64 > 128
        ])

        self.finales = nn.ModuleList([ finale(512), finale(256), finale(128), finale(64) ])

        meta.setdefault('exportResolutions', [16,32,64,128])
        self.exportResolutions = meta['exportResolutions']

        self.head.apply   (weights_init_kaiming2)
        #self.blocks.apply (weights_init_dcgan)
        #self.blocks.apply (weights_init_kaiming2)
        self.finales.apply(weights_init_dcgan)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)
        x = self.head(x)

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= 1: outs.insert(0, self.finales[i-1](x))

        return outs


'''

class MsgFirstG(nn.Module):
    def __init__(self, meta):
        super().__init__()

        upsample = Upsample2d()
        #upsample = Upsample2d(mode='bilinear')

        useBn = False
        useBn = True
        bn = nn.BatchNorm2d if useBn else lambda n: nn.Sequential()


        bias = False
        def first_block(cin,cout, bn=useBn):
            layers = [
                    #nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                    #*([nn.BatchNorm2d(cout)] if bn else []),
                    #ReLU(),
                    #nn.ConvTranspose2d(cout, cout, 4, 2, 1, bias=False),
                    nn.ConvTranspose2d(cin, cout, 4, 4, 0, bias=bias),
                    *([nn.BatchNorm2d(cout)] if bn else []),
                    ReLU(),
                    ]
            return nn.Sequential(*layers)
        def block(cin,cout, up=True, bn=useBn):
            #cmid = (cin + cout) // 2
            cmid = cout
            layers = [
                      *([upsample] if up else []), nn.Conv2d(cin, cmid, 3, 1, 1, bias=bias),
                      #nn.ConvTranspose2d(cin, cmid, 4, 2, 1, bias=bias),
                      *([nn.BatchNorm2d(cmid)] if bn else []),
                      ReLU(),
                      #nn.Conv2d(cmid, cout, 3, 1, 1, bias=bias),
                      #*([nn.BatchNorm2d(cout)] if bn else []),
                      #ReLU()
                      ]
            return nn.Sequential(*layers)
        def finale(cin, bn=useBn):
            #layers = [nn.Conv2d(cin,3, 1,1,0, bias=False)]
            #return nn.Sequential(*layers)
            return nn.Conv2d(cin,3, 1,1,0, bias=False)

        meta.setdefault('exportResolutions', [16,32,64,128])
        self.exportResolutions = meta['exportResolutions']

        blocks = [
            first_block(meta['latentSize'], 512), # 0) 1  > 4
            block(512, 512),                      # 1) 4  > 8
            block(512, 512),                      # 2) 8  > 16
            block(512, 256),                      # 3) 16 > 32
            block(256, 128),                      # 4) 32 > 64
            block(128, 128),                      # 5) 64 > 128
        ]
        finales = [ finale(512), finale(256), finale(128), finale(128) ]

        self.blocks  = nn.ModuleList(blocks)
        self.finales = nn.ModuleList(finales)

        self.blocks.apply (init_fn)
        self.finales.apply(init_fn)

    def forward(self, z):
        x = z.view(z.size(0), z.size(1), 1, 1)

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= 2: outs.insert(0, self.finales[i-2](x))

        return outs

'''

class Combine(nn.Module):
    def __init__(self):
        super().__init__()
        #self.scale = nn.Parameter(torch.ones((1,1,1,1),requires_grad=False)*.1, requires_grad=True)
        #self.bias = nn.Parameter(torch.zeros((1,1,1,1),requires_grad=False), requires_grad=True)
        #print(' - Note: adding noise in Combine()')
        #print(' - Note: Combine() only using first input.')
    def forward(self, x, y):
        #y = y + torch.randn_like(y) / 20
        #if x is None: return              y*self.scale+self.bias
        #else:         return torch.cat((x,y*self.scale+self.bias),1)
        if x is None: return              y
        else:         return torch.cat((x,y),1)
        #if x is None: return              y
        #else:         return torch.cat((x,torch.zeros_like(y)),1)


class BottleneckD(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.Sequential()):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class BasicBlockD(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, stats=True):
        super().__init__()
        if norm_layer is None: norm_layer = lambda x: nn.Sequential()
        if groups != 1 or base_width != 64: raise ValueError('BasicBlockD only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlockD")
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = norm_layer(planes)
        self.relu = ReLU()
        self.stats,cc = (MinibatchStdDev(),1) if stats else (nn.Sequential(),0)
        self.conv2 = conv3x3(planes+cc, planes)
        #self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.stats(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class MsgFirstD(nn.Module):
    def __init__(self, meta):
        super().__init__()

        Block = BasicBlockD

        def make_down(cin, cout, stride=2):
            return conv1x1(cin, cout*Block.expansion, stride=stride)

        #conv1x1(self.inplanes, planes * block.expansion, stride),
        self.blocks = nn.ModuleList([
            Block(3, 64, 2, make_down(3,64)),         # 256
            Block(64+3, 128, 2, make_down(64+3,128)),   # 128
            Block(128+3, 256, 2, make_down(128+3,256)), # 64
            Block(256+3, 512, 2, make_down(256+3,512)), # 32
            Block(512, 512, 2, make_down(512,512)),   # 16
            Block(512, 512, 2, make_down(512,512)),   # 8
            Block(512, 512, 2, make_down(512,512)),   # 4
        ])

        self.finale = nn.Sequential(
            *StdLayer(),
            nn.Conv2d(512+stdLayerD, 512, 1, bias=False),
            ReLU(),
            nn.Conv2d(512, 1, 1, bias=False)
        )

        self.combines = nn.ModuleList([Combine() for _ in range(4)])

        self.blocks.apply(weights_init_kaiming)
        self.finale.apply(weights_init_kaiming)

    def forward(self, xs):
        assert(len(xs) == 4)
        y = self.combines[0](None, xs[0])
        #y = None

        for i, blk in enumerate(self.blocks):
            y = blk(y)
            if i < 3: y = self.combines[i+1](y, xs[i+1])
            #y = blk(y, xs[i+1])
            #y = blk(y, xs[0])
            #if i < 3: y = self.combines[i+1](y, xs[i+1])

        y = self.finale(y)
        return y



class MsgResnetD(nn.Module):
    def __init__(self, meta):
        super().__init__()

        from torchvision.models import resnet50
        from torchvision.ops.misc import FrozenBatchNorm2d
        norm_layer = FrozenBatchNorm2d if meta['resnetFreezeBn'] else nn.BatchNorm2d
        cs = list(resnet50(True,norm_layer=FrozenBatchNorm2d).children())[:8]

        self.sides = nn.ModuleList([
            nn.Sequential(),
            nn.Conv2d(3, 64, 1, 1, bias=False),
            nn.Conv2d(3, 64, 1, 1, bias=False),
            nn.Conv2d(3, 512, 1, 1, bias=False)
        ])

        self.blocks = nn.ModuleList([
                nn.Sequential(*cs[0:3]),
                nn.Sequential(*cs[3:4]),
                nn.Sequential(*cs[4:6]),
                nn.Sequential(*cs[6: ]),
        ])

        self.finale = nn.Sequential(
                nn.Conv2d(2048,1,1,bias=False)
        )

    def forward(self, xs):
        x = self.sides[0](xs[0])
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if i < 3: x = x + self.sides[i+1](xs[i+1])
        return self.finale(x)

"""
dUseBn = True
dUseBn = False
dUseIn = False
dUsePn = False
dUseLn = False
dNormLayer = lambda c: nn.Sequential()
if dUseBn: dNormLayer = lambda c: nn.BatchNorm2d(c)
if dUseIn: dNormLayer = lambda c: nn.InstancedNorm2d(c, False)
if dUsePn: dNormLayer = lambda c: PixNorm()
if dUseLn: dNormLayer = lambda c: LayerNorm2d(c)

class MsgBlockD(nn.Module):
    def __init__(self, cin, cout, down=True, bn1=True, conv1=True, k=3):
        super().__init__()


        self.stats,extra = MinibatchStdDev(), 1

        # include second conv?
        if True and conv1:
            cmid = cin
            self.conv1 = nn.Conv2d(cin+extra, cmid, k, 1, k//2, bias=False)
            self.norm1 = dNormLayer(cmid) if bn1 else nn.Sequential()
            self.relu1 = ReLU()
        else:
            cmid = cin+extra
            self.conv1 = nn.Sequential()
            self.norm1 = nn.Sequential()
            self.relu1 = nn.Sequential()

        # pool or strided conv?
        if True:
            self.conv2 = nn.Conv2d(cmid,      cout, 3, 2, 3//2, bias=False)
            self.pool = nn.Sequential()
        else:
            self.conv2 = nn.Conv2d(cmid,      cout, 3, 1, 3//2, bias=False)
            self.pool = nn.AvgPool2d(2,2)
        self.norm2 = dNormLayer(cout)
        self.relu2 = ReLU()

    def forward(self, x):
        x = self.stats(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool(x)
        x = self.relu2(x)
        return x

# Note: this is convolutional and has spatial output dims.
#       the one in the paper is not.
class MsgFirstD(nn.Module):
    def __init__(self, meta):
        super().__init__()


        '''def block(cin, cout, down=True, bn1=useBn, k=3):
            #cmid = (cin + cout) // 2
            #cmid = cin + stdLayerD
            cmid = cin
            blk = nn.Sequential()
            blk.add_module('0',StdLayer())
            return nn.Sequential(*[
                *StdLayer(),
                ('blkConv1',nn.Conv2d(cin+stdLayerD, cmid, k, 1, k//2, bias=False)),
                *(dNormLayer(cmid) if bn1 else []),
                ReLU(),
                #nn.Conv2d(cmid, cout, 3, 1, 1, bias=False),
                (nn.Conv2d(cmid, cout, 3, 2, 1, bias=False) if down else nn.Conv2d(cmid, cout, 3, 2, 1, bias=False)),
                *dNormLayer(cout),
                #*([nn.AvgPool2d(2,2)] if down else []),
                ReLU(),
            ])'''
        block = MsgBlockD

        blocks = [
            #block(3, 32, k=5, bn=False),            # 0) 128 > 64
            block(3, 128, k=5, bn1=False,conv1=False),            # 0) 128 > 64
            #block(3, 128, k=3),       # 0) 128 > 64
            block(128+3, 128, bn1=False),         # 1) 64 > 32
            block(128+3, 256, bn1=False),        # 2) 32 > 16
            block(256+3, 256),       # 3) 16 > 8
            block(256, 512),         # 4) 8 > 4
            block(512, 512, False), # 5) 4 = 4
        ]

        self.blocks = nn.ModuleList(blocks)
        self.finale = nn.Sequential(
            *StdLayer(),
            nn.Conv2d(512+stdLayerD, 512, 1, bias=False),
            ReLU(),
            nn.Conv2d(512, 1, 1, bias=False)
        )
        self.combines = nn.ModuleList([Combine() for _ in range(4)])

        #self.blocks.apply(weights_init_dcgan)
        #self.finale.apply(weights_init_dcgan)

        #for n,p in self.named_modules():
            #if 'conv1' in n and hasattr(p,'weight'): p.apply(weights_init_diag)

    def forward(self, xs):
        assert(len(xs) == 4)
        y = self.combines[0](None, xs[0])

        for i, blk in enumerate(self.blocks):
            y = blk(y)
            if i < 3:
                y = self.combines[i+1](y, xs[i+1])

        y = self.finale(y)
        return y
"""

'''
Override gan-base since we need to do some special things
with intermediate layers that doesn't fit the typical GAN process.
'''
class MsgGan(GanBase):
    def __init__(self, meta):
        super().__init__(meta, skipCreation=True)

        self.net_g = MsgFirstG(meta)

        if meta['D'] == 'resnet':
            self.net_d = MsgResnetD(meta)
        else:
            self.net_d = MsgFirstD(meta)

        from gans.base import get_optimizers
        self.opt_d, self.opt_g = get_optimizers(self.net_d, self.net_g, meta)

    # Override forward so that we can use externally without problems.
    def forward_d(self, xs):
        if isinstance(xs, torch.Tensor):
            xs = [self.pix_normalizer(xs)]
            for _ in range(3):
                xs.append(F.avg_pool2d(xs[-1], 2,2))
        else:
            xs = [self.pix_normalizer(x) for x in xs]

        return self.net_d(xs)

    # Override penalty functions to handle lists
    def loss_gp(self, r,fs):
        xs = []
        for f in fs:
            t = torch.rand(r.size(0), 1, r.size(2), r.size(3), device=r.device)
            xs.append(torch.autograd.Variable(r.detach() * t + f.detach() * (1-t), True))
            r = F.avg_pool2d(r,2,2)

        pred_x = self.forward_d(xs)
        gradss = torch.autograd.grad(
                outputs=pred_x, inputs=xs,
                grad_outputs=torch.ones_like(pred_x.data),
                retain_graph=True,only_inputs=True,create_graph=True)
        ls = [((safe_norm(grads.view(grads.size(0),-1),dim=1) - 1) ** 2).mean() for grads in gradss]
        return sum(ls)

    def loss_dragan(self, r, k=1, sigma=.2):
        xs = []
        for i in range(4):
            xs.append(torch.autograd.Variable(r.detach()+torch.randn_like(r)*sigma, True))
            r = F.avg_pool2d(r,2,2).detach()
        pred_x = self.forward_d(xs)
        gradss = torch.autograd.grad(
                outputs=pred_x, inputs=xs,
                grad_outputs=torch.ones_like(pred_x.data),
                retain_graph=True,only_inputs=True,create_graph=True)
        ls = [((grads.view(grads.size(0),-1).norm(dim=1) - 1) ** 2).mean() for grads in gradss]
        return sum(ls)
