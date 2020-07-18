import torch, torch.nn.functional as F, torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os, sys

from gans.utils import PixelNormalization

def get_networks(meta):
    # Get D
    if meta['D'] == 'dcgan':
        from gans.models.dcgan import DcganDiscriminator
        d = DcganDiscriminator(meta)
    elif meta['D'] == 'resnet':
        from gans.models.resnet import ResnetD
        d = ResnetD(meta)
    else:
        raise ValueError('unknown D architecture: {}'.format(meta['D']))

    if meta['useSpectralNorm']: apply_spectral_norm(d)

    # Get G
    if meta['G'] == 'dcgan':
        from gans.models.dcgan import DcganGenerator
        g = DcganGenerator(meta)
    elif meta['G'] == 'stylegan':
        from gans.models.stylegan import StyleGanGenerator
        g = StyleGanGenerator(meta)
    else:
        raise ValueError('unknown G architecture: {}'.format(meta['G']))

    return d,g

def get_optimizers(net_d, net_g, meta):
    opt_d = torch.optim.Adam(net_d.parameters(), meta['lr_d'], betas=(.5,.999))
    opt_g = torch.optim.Adam(net_g.parameters(), meta['lr_g'], betas=(.5,.999))
    return opt_d, opt_g

def safe_norm(x, dim, eps=1e-7):
    return ((x**2).sum(dim=dim) + eps).sqrt()


class GanBase(nn.Module):
    def __init__(self, meta, skipCreation=False):
        super().__init__()
        meta.setdefault('name', 'unnamedGan')
        meta.setdefault('latentSize', 512)
        meta.setdefault('penalty', 'gp') # or dragan
        meta.setdefault('penaltyFactor', 10)
        meta.setdefault('loss', 'lsganTwo')
        meta.setdefault('useSpectralNorm', False)
        meta.setdefault('numChannels', 3)
        meta.setdefault('instanceNoise', 0)
        meta.setdefault('clampG', True)
        self.meta = meta
        for k,v in meta.items(): setattr(self, k, v)
        self.device = torch.device('cuda')

        self.pix_normalizer = PixelNormalization(meta['numChannels'])

        if not skipCreation:
            self.net_d, self.net_g = get_networks(meta)
            self.opt_d, self.opt_g = get_optimizers(self.net_d, self.net_g, meta)

    def forward(self, *a,**k):
        raise NotImplementedError('Call forward_g or forward_d')

    def forward_d(self, x):
        x = self.pix_normalizer(x)
        return self.net_d(x)

    def forward_g(self, z=None, batchSize=None):
        if z is None:
            z = torch.randn(batchSize, self.latentSize, device=self.device)
        if self.normalizeLatent: z = F.normalize(z)

        y = self.net_g(z)

        if self.clampG:
            if isinstance(y, list): y = [self.pix_normalizer.backward(yy).clamp(0,1) for yy in y]
            else: y = self.pix_normalizer.backward(y).clamp(0,1)
        else:
            if isinstance(y, list): y = [self.pix_normalizer.backward(yy) for yy in y]
            else: y = self.pix_normalizer.backward(y)
        return y


    def loss_d(self, real, fake=None, penaltyFactor=None):
        if fake is None: fake = self.forward_g(batchSize=real.size(0))
        if penaltyFactor is None: penaltyFactor = self.penaltyFactor

        if self.realFakeDifferentBatch:
            pred_r = self.forward_d(real)
            if self.dFakeEval: self.net_d.eval()
            pred_f = self.forward_d(fake)
            if self.dFakeEval: self.net_d.train()
        elif isinstance(fake, list):
            rs = [real]
            for _ in range(3): rs.append(F.avg_pool2d(rs[-1], 2,2))
            xs = [torch.cat((r,f),0) for r,f in zip(rs,fake)]
            pred_rf = self.forward_d(xs)
            pred_r = pred_rf[:real.size(0)]
            pred_f = pred_rf[real.size(0):]
        else:
            pred_rf = self.forward_d(torch.cat((real,fake),0))
            pred_r = pred_rf[:real.size(0)]
            pred_f = pred_rf[real.size(0):]

        inr,inf = 0, 0
        if self.instanceNoise > 0:
            inr = torch.randn(pred_r.size(0),device=pred_r.device).view(-1,1,1,1) * self.instanceNoise
            inf = torch.randn(pred_f.size(0),device=pred_f.device).view(-1,1,1,1) * self.instanceNoise
        if self.loss == 'lsganOne':
            loss_dr = ((pred_r-1+inr)**2).mean()
            loss_df = ((pred_f  +inf)**2).mean()
        elif self.loss == 'lsganTwo':
            loss_dr = ((pred_r-1+inr)**2).mean()
            loss_df = ((pred_f+1+inf)**2).mean()
        elif self.loss == 'wgan':
            loss_dr = -pred_r.mean()
            loss_df =  pred_f.mean()
        elif self.loss == 'ns':
            #loss_dr = F.binary_cross_entropy_with_logits(pred_r, torch.ones_like(pred_r))
            #loss_df = F.binary_cross_entropy_with_logits(pred_r, torch.ones_like(pred_r))
            loss_dr = -F.logsigmoid(pred_r).mean()
            loss_df = -F.logsigmoid(1 - pred_f).mean()

        loss = loss_dr + loss_df
        losses = dict(dr=loss_dr.detach(), df=loss_df.detach())

        if penaltyFactor>0 and self.penalty == 'gp':
            loss_penalty = self.loss_gp(real, fake) * penaltyFactor
            loss = loss + loss_penalty
            losses['gp'] = loss_penalty.detach()
        if penaltyFactor>0 and self.penalty == 'dragan':
            loss_penalty = self.loss_dragan(real) * penaltyFactor
            loss = loss + loss_penalty
            losses['dragan'] = loss_penalty.detach()
        if penaltyFactor>0 and self.penalty == 'sgp':
            loss_penalty = self.loss_sgp(real) * penaltyFactor
            loss = loss + loss_penalty
            losses['sgp_r'] = loss_penalty.detach()
            #loss_penalty = self.loss_sgp(fake) * penaltyFactor
            #loss = loss + loss_penalty
            #losses['sgp_f'] = loss_penalty.detach()

        return loss, losses

    def loss_g(self, imgs, secondHalf=False):
        pred = self.forward_d(imgs)
        if secondHalf: pred = pred[pred.size(0)//2:]

        if self.loss == 'lsganOne':
            loss = ((pred-1)**2).mean()
        elif self.loss == 'lsganTwo':
            loss = ((pred)**2).mean()
        elif self.loss == 'wgan':
            loss = -pred.mean()
        elif self.loss == 'ns':
            loss = -F.logsigmoid(pred).mean()

        return loss, dict(g=loss.detach())


    # TODO: the view thing doesn't make sense... there is no batch dim, should just be norm()
    # or does it...
    def loss_gp(self, r,f):
        t = torch.rand(r.size(0), 1, r.size(2), r.size(3), device=r.device)
        x = torch.autograd.Variable(r.detach() * t + f.detach() * (1-t), True)
        pred_x = self.forward_d(x)
        grads = torch.autograd.grad(
                outputs=pred_x, inputs=x,
                grad_outputs=torch.ones_like(pred_x.data),
                retain_graph=True,only_inputs=True,create_graph=True)[0]
        grads = grads.view(grads.size(0), -1)
        l = ((safe_norm(grads,dim=1) - 1) ** 2).mean()
        return l

    def loss_sgp(self, r, sigma=.01):
        x = torch.autograd.Variable(r.detach()+torch.randn_like(r)*sigma, True)
        pred_x = self.forward_d(x)
        grads = torch.autograd.grad(
                outputs=pred_x, inputs=x,
                grad_outputs=torch.ones_like(pred_x.data),
                retain_graph=True,create_graph=True)
        #l = sum([g.norm() for g in grads]) / len(grads)
        l = sum([(g**2).view(g.size(0),-1).sum(dim=1).mean() for g in grads]) / len(grads)
        return l


    def loss_dragan(self, r, k=1, sigma=.2):
        x = torch.autograd.Variable(r.detach()+torch.randn_like(r)*sigma, True)
        pred_x = self.forward_d(x)
        grads = torch.autograd.grad(
                outputs=pred_x, inputs=x,
                grad_outputs=torch.ones_like(pred_x.data),
                retain_graph=True,only_inputs=True,create_graph=True)[0]
        grads = grads.view(grads.size(0), -1)
        l = ((grads.norm(dim=1) - k) ** 2).mean()
        return l





    def save(self, epoch=-1, dir='saves/gan'):
        if epoch != -1: fname = os.path.join(dir, self.name + '.{}'.format(epoch) + '.pt')
        else: fname = os.path.join(dir, self.name + '.pt')
        print(' - saving', fname)
        torch.save({
            'meta': self.meta,
            'sd': self.state_dict(),
            'name': self.name,
            'type': self.__class__.__name__,
            'opt_d_sd': self.opt_d.state_dict() if self.restoreOpt else None,
            'opt_g_sd': self.opt_g.state_dict() if self.restoreOpt else None,
            'epoch': epoch}, fname)

    @staticmethod
    def load(path, restoreOpt=False, **kw):
        d = torch.load(path)
        meta = None
        if 'meta' in d:
            meta = d['meta']
        else:
            for k,v in d.items():
                try:
                    if 'meta' in v:
                        meta = v['meta']
                        break
                except: pass
        assert(meta is not None)


        clazz = d['type']
        for k,v in kw.items(): meta[k] = v
        from gans.models.msggan import MsgGan
        globals()['MsgGan'] = MsgGan
        model = globals()[clazz](meta).cuda()
        model.load_state_dict(d['sd'])
        if restoreOpt:
            model.opt_d.load_state_dict(d['opt_d_sd'])
            model.opt_g.load_state_dict(d['opt_g_sd'])
        return model,d


    def __repr__(self):
        s  = '\n============= GanBase ============'
        s += '\n - loss: ' + self.loss
        s += '\n - penalty: ' + self.penalty
        s += '\n - D'
        s += '\n' + self.net_d.__repr__()
        s += '\n - G'
        s += '\n' + self.net_g.__repr__()
        s += '\n==================================\n'
        return s

    '''
    # Some hackery to save state dict params.
    def _load_from_state_dict(self, sd, *a, **kw):
        sd_od, sd_og = {}, {}
        to_del = []
        for k,v in sd.items():
            if 'opt_d.' in k:
                sd_od[k.replace('opt_d.','')] = v
                to_del.append(k)
            if 'opt_g.' in k:
                sd_og[k.replace('opt_g.','')] = v
                to_del.append(k)
        for k in to_del: del sd[k]
        self.opt_d.load_state_dict(sd_od)
        self.opt_g.load_state_dict(sd_og)
        return super()._load_from_state_dict(sd, *a, **kw)
    def state_dict(self):
        sd0_,sd0 = self.opt_d.state_dict(), {}
        sd1_,sd1 = self.opt_g.state_dict(), {}
        for k,v in sd0_.items(): sd0['opt_d.'+k] = v
        for k,v in sd1_.items(): sd1['opt_g.'+k] = v
        sd2 = super().state_dict()
        sd2.update(sd0)
        sd2.update(sd1)
        return sd2
    '''
