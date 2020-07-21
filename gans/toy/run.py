from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import cv2,sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from ..utils import MinibatchStdDev, MinibatchStdDevWide, MinibatchStdDevWideSpatial

from gans.datasets.folder_datasets import DataLoader, FoldersDataset#, RecursiveFoldersDataset

class Upsample2d(nn.Module):
    #def __init__(self, factor=2, mode='bilinear'):
    def __init__(self, factor=2, mode='nearest'):
        super().__init__()
        self.factor, self.mode = factor, mode
    def forward(self, x):
        return F.interpolate(x,scale_factor=self.factor, mode=self.mode)

def weights_init_mine(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.orthogonal_(m.weight)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            torch.nn.init.zeros_(m.bias)
def weights_init_dcgan(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

def plot_losses(l):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig = plt.gcf()
    for k,v in l.items():
        ax.plot(v, label=k)
    fig.legend()
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    d = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    d = d.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    cv2.imshow('losses', cv2.cvtColor(d,cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)
    plt.close(fig)
    del fig

def show_imgs(r,f):
    r = r.mul(255).permute(0,2,3,1).to(torch.uint8).detach().cpu().numpy()
    f = f.mul(255).permute(0,2,3,1).to(torch.uint8).detach().cpu().numpy()
    r = np.concatenate(r, 1)
    f = np.concatenate(f, 1)
    img = np.concatenate((r,f), 0)
    cv2.imshow('imgs',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


instanceNoise = .2 # stddevs
instanceNoise = 0

def normalize(x): return (x-.45)/.25
def unnormalize(x): return ((x*.25)+.45).clamp_(0,1)

def get_loss_d(d, g, r, f):

    # Instance noise.
    if instanceNoise>0:
        r = r + torch.randn_like(r) * instanceNoise
        f = f + torch.randn_like(f) * instanceNoise

    #dr = -d(r).mean()
    #df = d(f).mean()
    dr,df = d(r), d(f)
    dr = F.binary_cross_entropy_with_logits(dr, torch.ones_like(dr))
    df = F.binary_cross_entropy_with_logits(df, torch.zeros_like(df))
    #dr = -F.logsigmoid(dr).mean()
    #df = -F.logsigmoid(1 - df).mean()

    loss = (dr + df) / 1
    losses = dict(dr=dr.item(),df=df.item())
    loss.backward()

    if True:
        #r = torch.autograd.Variable((r+f.detach())/2,True)
        r = torch.autograd.Variable(r,True)
        #d_on_real = torch.sigmoid(d(r))
        d_on_real = (d(r))
        grads = torch.autograd.grad(
            #outputs=d_on_real, inputs=d.parameters(),
            outputs=d_on_real.sum(), inputs=r,
            create_graph=True,retain_graph=True,only_inputs=True)

        sgp = grads[0].pow(2).view(r.size(0),-1).sum(1)
        sgp = sgp.mean() * 5

        losses['sgp'] = sgp.item()
        loss = loss + sgp
        sgp.backward()

    if False:
        z = torch.randn(f.size(0),512,1,1, requires_grad=True, device=f.device)
        ff = g(z)
        d_on_ff = (d(ff))
        grad = torch.autograd.grad(
            outputs=d_on_ff.sum(), inputs=z,
            create_graph=True,retain_graph=True,only_inputs=True)[0]
        lf = (grad.pow(2).view(z.size(0), -1).sum(1).sqrt() - 1).pow(2).mean()
        losses['lf'] = lf
        lf.backward()



    if False:
        r = torch.autograd.Variable(r,True)
        d_on_real = d(r+torch.randn_like(r)/9)
        #d_on_real = torch.sigmoid(d_on_real) # Do, or no??? dragan was written for wgan.
        dragan = torch.autograd.grad(
            outputs=d_on_real, inputs=r,
            grad_outputs=torch.ones_like(d_on_real),
            create_graph=True,only_inputs=True,retain_graph=True)[0]
        # dragan like gp.
        #dragan = sum([((((g.view(g.size(0),-1))**2).sum(1).sqrt()-1)**2).mean() for g in dragan]) / len(dragan)
        #dragan = (((((dragan/r.size(0)))**2).sum().sqrt()-1)**2).mean()
        dragan = ((dragan/r.size(0)).norm()-1)**2
        dragan = dragan * 1
        losses['dragan'] = dragan.item()
        loss = loss + dragan
        dragan.backward()

    return loss, losses
def get_loss_g(d, g, f):
    if instanceNoise>0: f = f + torch.randn_like(f) * instanceNoise
    #g = -d(f).mean()
    #g = -F.logsigmoid(d(f)).mean()
    df = d(f)
    loss = F.binary_cross_entropy_with_logits(df, torch.ones_like(df))
    loss.backward()
    losses = dict(g=loss.item())

    if False:
        z = torch.randn(f.size(0), 512, 1, 1, requires_grad=True, device=f.device)
        ff = g(z)
        grads = torch.autograd.grad(
            #outputs=d_on_real, inputs=d.parameters(),
            outputs=ff.mean(), inputs=z,
            create_graph=True,retain_graph=True,only_inputs=True)
        #lf = grads[0].pow(2).view(z.size(0), -1).sum(1).mean()
        lf = (grads[0].pow(2).view(z.size(0), -1).sum(1).sqrt()-1).pow(2).mean()
        lf.backward()
        loss = loss + lf
        losses['lf'] = lf.item()

    return loss, losses

def next_data(diter, dev):
    #return F.avg_pool2d(next(diter).to(dev), (3,3))
    return next(diter).to(dev)

def main():
    resolution = 128
    batchSize = 72*3//2
    latentDims = 512
    numUpdatesD, numUpdatesG = 1, 1

    print(' - Using Dataset: CelebA')
    cats = ['/data/celeba/']
    dset = FoldersDataset(cats, crop='center', img_size=resolution)

    loader = DataLoader(dset, batch_size=batchSize, num_workers=2, shuffle=False)
    diter = iter(loader)

    normLayer = nn.BatchNorm2d
    #normLayer = lambda c: nn.Sequential()
    #normLayer = lambda c: nn.Sequential()
    #act = lambda: nn.ReLU(True)
    act = lambda: nn.LeakyReLU(.2, True)
    statsLayer, extra = MinibatchStdDev, 1
    statsLayer, extra = lambda: MinibatchStdDevWideSpatial(4,16), 16
    #statsLayer, extra = lambda: nn.Sequential(), 0

    upsample = Upsample2d()

    s = 2
    '''
    net_g = nn.Sequential(
        # StyleGan-like latent mapping
        nn.Conv2d(latentDims, latentDims, 1,1,0),
        nn.ReLU(True),
        #nn.Conv2d(latentDims, latentDims, 1,1,0),
        #nn.ReLU(True),
        #nn.Conv2d(latentDims, latentDims, 1,1,0),
        #nn.ReLU(True),

        nn.ConvTranspose2d(latentDims, 512, 4,2,1, bias=False), # 2
        normLayer(512),
        act(),
        upsample,
        nn.Conv2d(512, 256, 3,1,1, bias=False), # 4
        normLayer(256),
        act(),
        upsample,
        nn.Conv2d(256, 128, 3,1,1, bias=False), # 8
        normLayer(128),
        act(),
        upsample,
        nn.Conv2d(128, 64, 3,1,1, bias=False), # 16
        normLayer(64),
        act(),
        upsample,
        nn.Conv2d(64, 32, 3,1,1, bias=False), # 32
        normLayer(32),
        act(),
        upsample,
        nn.Conv2d(32, 32, 3,1,1, bias=False), # 64
        normLayer(32),
        act(),
        upsample,
        nn.Conv2d(32, 3, 3,1,1, bias=False), # 128
    )
    '''
    '''
    net_d = nn.Sequential(
        nn.Conv2d(3, 32, 3, s, 1, bias=False), # 128
        act(), statsLayer(),
        nn.Conv2d(32+extra, 64, 3, s, 1, bias=False), # 64
        normLayer(64), act(), statsLayer(),
        nn.Conv2d(64+extra, 64, 3, s, 1, bias=False), # 32
        normLayer(64), act(), statsLayer(),
        nn.Conv2d(64+extra, 128, 3, s, 1, bias=False), # 16
        normLayer(128), act(), statsLayer(),
        nn.Conv2d(128+extra, 256, 3, s, 1, bias=False), # 8
        normLayer(256), act(), statsLayer(),
        nn.Conv2d(256+extra, 256, 3, s, 1, bias=False), # 4
        normLayer(256), act(), statsLayer(),
        nn.Conv2d(256+extra, 512, 3, s, 1, bias=False), # 2
        normLayer(512), act(), statsLayer(),
        nn.Conv2d(512+extra, 512, 3, s, 1, bias=False), # 1
        normLayer(512), act(),
        nn.Conv2d(512, 1, 1, bias=False)
    )
    '''
    net_d = nn.Sequential(
        nn.Conv2d(3, 64, 3, s, 1, bias=False), # 128
        act(), MinibatchStdDevWideSpatial(4,8),
        nn.Conv2d(64+8, 64, 3, s, 1, bias=False), # 64
        normLayer(64), act(), MinibatchStdDevWideSpatial(4,16),
        nn.Conv2d(64+16, 64, 3, s, 1, bias=False), # 32
        normLayer(64), act(), MinibatchStdDevWideSpatial(4,16),
        nn.Conv2d(64+16, 128, 3, s, 1, bias=False), # 16
        normLayer(128), act(), MinibatchStdDevWideSpatial(4,16),
        nn.Conv2d(128+16, 256, 3, s, 1, bias=False), # 8
        normLayer(256), act(), MinibatchStdDevWide(4,16),
        nn.Conv2d(256+16, 256, 3, s, 1, bias=False), # 4
        normLayer(256), act(), MinibatchStdDevWide(4,16),
        nn.Conv2d(256+16, 512, 3, s, 1, bias=False), # 2
        normLayer(512), act(), MinibatchStdDevWide(4,16),
        nn.Conv2d(512+16, 512, 3, s, 1, bias=False), # 1
        normLayer(512), act(),
        nn.Conv2d(512, 1, 1, bias=False)
    )
    net_g = nn.Sequential(
        # StyleGan-like latent mapping
        #nn.Conv2d(latentDims, latentDims, 1,1,0),
        #nn.ReLU(True),
        #nn.Conv2d(latentDims, latentDims, 1,1,0),
        #nn.ReLU(True),
        #nn.Conv2d(latentDims, latentDims, 1,1,0),
        #nn.ReLU(True),

        nn.ConvTranspose2d(latentDims, 512, 4,2,1, bias=False), # 2
        normLayer(512), act(), upsample,
        nn.Conv2d(512, 256, 3,1,1, bias=False), # 4
        normLayer(256), act(),
        nn.Conv2d(256, 256, 3,1,1, bias=False), # 4
        normLayer(256), act(), upsample,
        nn.Conv2d(256, 128, 3,1,1, bias=False), # 8
        normLayer(128), act(),
        nn.Conv2d(128, 128, 3,1,1, bias=False), # 8
        normLayer(128), act(), upsample,
        nn.Conv2d(128, 64, 3,1,1, bias=False), # 16
        normLayer(64), act(),
        nn.Conv2d(64, 64, 3,1,1, bias=False), # 16
        normLayer(64), act(), upsample,
        nn.Conv2d(64, 32, 3,1,1, bias=False), # 32
        normLayer(32), act(),
        nn.Conv2d(32, 32, 3,1,1, bias=False), # 32
        normLayer(32), act(), upsample,
        nn.Conv2d(32, 32, 3,1,1, bias=False), # 64
        #normLayer(32), act(), upsample,
        #nn.Conv2d(32, 3, 3,1,1, bias=False), # 128
        act(),
        nn.ConvTranspose2d(32, 3, 4,2,1, bias=False), # 128
    )

    dev = torch.device('cuda')
    net_g.to(dev)
    net_d.to(dev)

    net_g.apply(weights_init_dcgan); net_d.apply(weights_init_dcgan)
    #net_g.apply(weights_init_mine); net_d.apply(weights_init_mine)

    lr = 5e-4
    #lr = 3e-3
    #lr = 7e-4
    #lr = 1e-3
    lr = 1e-4
    #opt_g = torch.optim.Adam(net_g.parameters(), lr=lr, betas=(.5,.999))
    #opt_d = torch.optim.Adam(net_d.parameters(), lr=lr, betas=(.5,.999))
    opt_g = torch.optim.RMSprop(net_g.parameters(), lr=lr)
    opt_d = torch.optim.RMSprop(net_d.parameters(), lr=lr)

    allLosses = defaultdict(lambda: [])
    allRunningLosses = defaultdict(lambda: 0)

    for i in range(10000):
        #net_g.eval()
        net_g.train()
        net_d.train()

        for j in range(numUpdatesD):
            with torch.no_grad():
                z = torch.randn(batchSize, latentDims, 1, 1, device=dev)
                f = net_g(z)
                r = normalize(next_data(diter,dev))


            loss,losses = get_loss_d(net_d, net_g, r, f)
            for k,v in losses.items(): allRunningLosses[k] = allRunningLosses[k] * .95 + v*.05
            if i > 26 and i % 25 == 0:
                for k,v in allRunningLosses.items(): allLosses[k].append(min(10,v))

            #loss.backward()
            opt_d.step()
            opt_g.step()
            opt_d.zero_grad()

        opt_g.zero_grad()
        net_g.train()
        net_d.train()
        for j in range(numUpdatesG):
            z = torch.randn(batchSize, latentDims, 1, 1, device=dev)
            f = net_g(z)

            loss,losses = get_loss_g(net_d, net_g, f)
            for k,v in losses.items(): allRunningLosses[k] = allRunningLosses[k] * .95 + v*.05
            if i > 26 and i % 25 == 0:
                for k,v in allRunningLosses.items(): allLosses[k].append(min(10,v))

            #loss.backward()
            opt_g.step()
            opt_d.zero_grad()
            opt_g.zero_grad()

        if i % 10 == 0:
            with torch.no_grad():
                net_g.eval()
                z = torch.randn(16, latentDims, 1, 1, device=dev)
                f = net_g(z)
                net_g.train()
                r = r + torch.randn_like(r)*instanceNoise

                plot_losses(allLosses)
                show_imgs(unnormalize(r[:16]), unnormalize(f[:16]))
                print(' -',i,':',end='')
                for k,v in allRunningLosses.items(): print('({}: {}) '.format(k,v),end='')
                print(' ')




main()
