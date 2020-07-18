import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import cv2

def weight_clip_func_1(l):
    with torch.no_grad():
        if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear) or isinstance(l, nn.BatchNorm2d):
            if hasattr(l,'bias') and l.bias is not None: l.bias.data.clamp_(-1,1)
            if hasattr(l,'weight') and l.weight is not None: l.weight.data.clamp_(-1,1)
def weight_clip_func_point_oh_one(l):
    with torch.no_grad():
        if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear) or isinstance(l, nn.BatchNorm2d):
            if hasattr(l,'bias') and l.bias is not None: l.bias.data.clamp_(-.01,.01)
            if hasattr(l,'weight') and l.weight is not None: l.weight.data.clamp_(-.01,.01)

def weights_init_diag(m):
    with torch.no_grad():
        torch.nn.init.zeros_(m.weight)
        k = m.weight.size(2)
        torch.nn.init.eye_(m.weight[:,:,k//2,k//2])
        print('diag init, result spatial means:', m.weight.mean(0).mean(0))

def latent_init(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        #assert('Linear' in classname or 'ReLU' in classname)
        if 'Linear' in classname:
            torch.nn.init.eye_(m.weight.data)
def weights_init_dcgan(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)
def weights_init_he(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            fan_in = np.prod(m.weight.shape)
            std = np.sqrt(2) / np.sqrt(fan_in)
            torch.nn.init.normal_(m.weight, 0.0, std)
def weights_init_kaiming(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #torch.nn.init.normal_(m.weight, 0.0, 0.002)
            torch.nn.init.kaiming_normal_(m.weight)
def weights_init_kaiming2(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #torch.nn.init.normal_(m.weight, 0.0, 0.002)
            torch.nn.init.kaiming_uniform_(m.weight)
            #torch.nn.init.xavier_normal_(m.weight,.1)

def apply_spectral_norm(m):
    classname = m.__class__.__name__
    if 'onv' in classname and hasattr(m,'weight') and m.weight is not None:
        print(' - applying sn to',m.weight.shape)
        torch.nn.utils.spectral_norm(m)

def adain_with_stats(x, mu, std):
    mu = mu.view(mu.size(0), mu.size(1), 1, 1)
    std = std.view(std.size(0), std.size(1), 1, 1)
    return F.instance_norm(x) * std + mu

class PixelNormalization(nn.Module):
    def __init__(self, numChannels):
        super().__init__()
        if numChannels == 3:
            mu = torch.FloatTensor([.485,.456,.406]).unsqueeze_(0).unsqueeze_(2).unsqueeze_(2).detach()
            sig = torch.FloatTensor([.229,.224,.225]).unsqueeze_(0).unsqueeze_(2).unsqueeze_(2).detach()
        else:
            mu = torch.FloatTensor([.4]).unsqueeze_(0).unsqueeze_(2).unsqueeze_(2).detach()
            sig = torch.FloatTensor([.3]).unsqueeze_(0).unsqueeze_(2).unsqueeze_(2).detach()
        self.register_buffer('mu', mu)
        self.register_buffer('sig', sig)

        #print('\n NOTE: PixelNormalization is currently inactive!\n')

    # Copy input, but then do second op in place.
    def forward(self, x):
        return x.sub(self.mu).div_(self.sig)
        return x
    def backward(self, x):
        return x.mul(self.sig).add_(self.mu)
        return x

class PixNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x / ((x*x).sum(dim=1, keepdim=True) + 1e-8).sqrt()
        #return x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + 1e-8)
        #return x-x.view(x.size(0),x.size(1),-1)mean(dim=2,keepdim=True) * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + 1e-8)
        #xx = x.view(x.size(0),x.size(1), -1)
        #return x * torch.rsqrt( (xx*xx).mean(dim=1) ).view(x.size(0),1,x.size(2),x.size(3))
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dims, aff=False):
        super().__init__(dims, 1e-5, aff)
    def forward(self, x):
        b,c,h,w = x.size()
        return x.sub(x.mean(1,keepdim=True)).div(x.std(1,keepdim=True))
        #return super().forward(x.permute(0,2,3,1).reshape(-1,c)).view(b,h,w,c).permute(0,3,1,2)

# https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L127
# I had to re-arrange the shapes a little (use dim 1, not 0)
class MinibatchStdDev(nn.Module):
    def __init__(self, groupSize=4):
        self.groupSize=4
        super().__init__()
    def forward(self, x):
        b,c,h,w = x.size()
        m,g = b//self.groupSize, self.groupSize
        y = x.view(m, g, c, h, w)
        y = y - y.mean(dim=1,keepdims=True)
        y = ((y**2).mean(1) + 1e-7).sqrt()
        y = y.view(m, -1).mean(dim=1)
        y = y.repeat_interleave(g).view(b, 1, 1, 1).repeat(1, 1, h, w)
        #return y
        return torch.cat( (x,y) , 1 )
class MinibatchStdDevWide(nn.Module):
    def __init__(self, groupSize=4, numChannels=32):
        self.groupSize=groupSize
        self.numChannels=numChannels
        super().__init__()
    def forward(self, x):
        b,c,h,w = x.size()
        m,g = b//self.groupSize, self.groupSize
        y = x.permute(0,2,3,1).view(m, g, h, w, c)
        y = y - y.mean(dim=1,keepdims=True)
        y = ((y**2).mean(1) + 1e-7).sqrt()
        y = y.permute(0,3,1,2).reshape(m, self.numChannels, -1).mean(dim=2)
        y = y.repeat_interleave(g,0).view(b, self.numChannels, 1, 1).repeat(1, 1, h, w)
        #return y
        return torch.cat( (x,y) , 1 )

class Upsample2d(nn.Module):
    #def __init__(self, factor=2, mode='bilinear'):
    def __init__(self, factor=2, mode='nearest'):
        super().__init__()
        self.factor, self.mode = factor, mode
    def forward(self, x):
        return F.interpolate(x,scale_factor=self.factor, mode=self.mode)

class NoiseAdder(nn.Module):
    def __init__(self, c, iniScale):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((1,c,1,1))*iniScale)
    def forward(self, x):
        b,c,h,w = x.size()
        return x.add_(torch.randn(b,1,h,w, device=x.device).mul(self.scale.repeat(b,1,h,w)))
class NoiseAdderRaw(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        b,c,h,w = x.size()
        return x.add(torch.randn_like(x).mul_(self.scale))




def viz_batch(model, realImgs, fakeImgs, epoch, showScores=True, dir='out/gans/'):
    with torch.no_grad():
        #model.eval()

        b,c,h,w = realImgs.size()
        extra = 0 if abs(b-int(np.sqrt(b))) < .0001 else 1
        rows = int(np.sqrt(fakeImgs.shape[0]) + extra)
        cols = int(np.sqrt(fakeImgs.shape[0]))
        dimg = np.zeros((rows*h,cols*w*2,3), dtype=np.uint8)
        real = realImgs.permute(0,2,3,1).mul(255).clamp(0,255).to(torch.uint8).detach().cpu().numpy()
        fake = fakeImgs.permute(0,2,3,1).mul(255).clamp(0,255).to(torch.uint8).detach().cpu().numpy()

        # Get scores.
        if showScores:
            x = torch.cat((realImgs,fakeImgs),0)
            scores = model.forward_d(x).detach().cpu().view(2,b,-1).mean(-1).permute(1,0)
            if model.loss == 'ns': scores = torch.sigmoid(scores)

        for i in range(b):
            y,x = i // rows, i % cols
            dimg[y*h:(y+1)*h,          x*w:         (x+1)*w] = real[i]
            dimg[y*h:(y+1)*h, cols*w + x*w:cols*w + (x+1)*w] = fake[i]
            if showScores:
                cv2.putText(dimg, '{:.3}'.format(scores[i,0]), (       x*w+10,y*h+20), cv2.FONT_HERSHEY_SIMPLEX,.7,(0,155,255))
                cv2.putText(dimg, '{:.3}'.format(scores[i,1]), (cols*w+x*w+10,y*h+20), cv2.FONT_HERSHEY_SIMPLEX,.7,(0,155,255))

        if dimg.shape[-1] == 3: dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('{}/tr.{:06d}.jpg'.format(dir,epoch), dimg)
        model.train()
        if showScores: del scores
        del realImgs, fakeImgs, dimg, x, real, fake#, fakeImgs#1, fakeImgs2

# MipMap-like layout
def viz_msg_batch(model, realImgs, fakeImgss, epoch, showScores=True, dir='out/gans/'):
    with torch.no_grad():
        #fakes = fakeImgss
        #reals = []
        fakeImgss = [f[:12] for f in fakeImgss]
        realImgss = [realImgs[:12]]
        for _ in range(len(fakeImgss)-1): realImgss.append(F.avg_pool2d(realImgss[-1],2,2))

        l = len(fakeImgss)
        n = fakeImgss[0].size(0)
        w = fakeImgss[0].size(-1) * n * 2
        h = fakeImgss[0].size(-2) * 3 // 2
        #for i in range(l): h += fakeImgss[i].size(-2)
        dimg = np.zeros((h,w,3), dtype=np.uint8)

        reals = [r.permute(0,2,3,1).mul(255).clamp(0,255).to(torch.uint8).detach().cpu().numpy() for r in realImgss]
        fakes = [f.permute(0,2,3,1).mul(255).clamp(0,255).to(torch.uint8).detach().cpu().numpy() for f in fakeImgss]

        # Get scores.
        if showScores:
            scores_r = model.forward_d(realImgss).detach().cpu().view(n,-1).mean(-1)
            scores_f = model.forward_d(fakeImgss).detach().cpu().view(n,-1).mean(-1)

        base_w = fakeImgss[0].size(-1)
        ooy, oox = 0, 0
        for i in range(l):
            ox, oy = 0, 0
            b1,c1,h1,w1 = fakeImgss[i].size()
            for j in range(b1):
                xx = ox + oox
                yy = oy + ooy
                dimg[yy:yy+h1, xx:xx+w1] = reals[i][j]
                dimg[yy:yy+h1, w//2+xx:w//2+xx+w1] = fakes[i][j]
                if showScores and i == 0:
                    cv2.putText(dimg, '{:.3}'.format(scores_r[j]), (xx+15,      20), cv2.FONT_HERSHEY_SIMPLEX,.5,(55,190,255))
                    cv2.putText(dimg, '{:.3}'.format(scores_f[j]), (xx+15+w//2, 20), cv2.FONT_HERSHEY_SIMPLEX,.5,(55,190,255))

                ox += base_w

            if i % 2 == 0: ooy += h1
            else: oox += w1


        if dimg.shape[-1] == 3: dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB)
        cv2.imwrite('{}/tr.{:06d}.jpg'.format(dir,epoch), dimg)
        #model.train()
