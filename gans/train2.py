import torch, torch.nn.functional as F, torch.nn as nn
import matplotlib
matplotlib.use('agg') # default gtk3 has trouble with my current opencv installation (gtk2)
import matplotlib.pyplot as plt
import numpy as np
import cv2

from gans.base import GanBase
from gans.utils import viz_batch, viz_msg_batch

makeBool = lambda x: x == '1' or x == 'y' or x == 1 or x == 'on'

def get_real_imgs(diter):
    if diter is None: return None
    try:
        realImgs = next(diter)
    except StopIteration:
        return None

    realImgs = realImgs.cuda()
    if realImgs.ndimension() == 6:
        bb,b,n,c,h,w = realImgs.size()
        realImgs = realImgs.view(bb*b,n,c,h,w)
    return realImgs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--latentSize', default=512)
    parser.add_argument('--numLatentMaps', default=8)
    parser.add_argument('--load', default=None)
    parser.add_argument('--restoreOpt', action='store_true')
    parser.add_argument('--saveEvery', default=1000,type=int)
    parser.add_argument('--batchSize', default=24,type=int)
    parser.add_argument('--numWorkers', default=3,type=int)
    #def_lr = 2e-4
    #def_lr = 1e-3
    #def_lr = 1e-2
    #def_lr = 3e-5
    #def_lr = 3e-3 # msg gan
    #def_lr = 8e-5 # msg gan
    #parser.add_argument('--lr_d', default=def_lr,type=float)
    #parser.add_argument('--lr_g', default=def_lr,type=float)
    #parser.add_argument('--lr', default=def_lr,type=float)
    parser.add_argument('--lr', required=True,type=float)
    parser.add_argument('--numUpdatesD', default=4,type=int)
    parser.add_argument('--numUpdatesG', default=1,type=int)
    parser.add_argument('--resolution', default=128,type=int)
    parser.add_argument('--D', default='dcgan')
    parser.add_argument('--G', default='dcgan')
    parser.add_argument('--penalty', default='gp')
    parser.add_argument('--penaltyFactor', default=10,type=float)
    parser.add_argument('--normalizeLatent', default=False,type=makeBool)
    parser.add_argument('--styleganLatentBatchNorm', default=False,type=makeBool)
    parser.add_argument('--resnetFreezeBn', default=True,type=makeBool)
    parser.add_argument('--addNoiseToD', default=True,type=makeBool)
    parser.add_argument('--loss', default='lsganTwo')
    parser.add_argument('--name', default='unnamedGan')
    parser.add_argument('--useMinibatchStd', default=False,type=makeBool)
    parser.add_argument('--msgGan', action='store_true')
    parser.add_argument('--sched', default=None)
    parser.add_argument('--realFakeDifferentBatch', default=True,type=makeBool)
    args = parser.parse_args()
    args.__dict__['lr_d'] = args.lr
    args.__dict__['lr_g'] = args.lr
    print(args)
    #if args.name is None: args.name='G.{}.D.{}.ls.{}'.format(args.G.lower(),args.D.lower(),args.

    # Construct model, either a fresh one or a checkpointed one
    if args.load is None:
        meta = dict(args.__dict__)
        if args.msgGan:
            from gans.models.msggan import MsgGan
            model = MsgGan(meta).train().cuda()
        else:
            model = GanBase(meta).train().cuda()
        ii = 0
    else:
        model, d = GanBase.load(args.load, restoreOpt=args.restoreOpt)
        model.train().cuda()
        ii = d['epoch'] + 1

    print(model)

    model.meta['lr_d'] = model.lr_d = args.__dict__['lr_d']
    model.meta['lr_g'] = model.lr_g = args.__dict__['lr_g']
    model.meta['loss'] = model.loss = args.loss
    model.meta['penaltyFactor'] = model.penaltyFactor = args.penaltyFactor
    model.meta['penalty'] = model.penalty = args.penalty

    loss_history, losses_ = {}, {}
    saveEvery = args.saveEvery
    printEvery = 25
    vizEvery = 100
    vizEvery = 5
    #vizEvery = 2

    from gans.datasets.folder_datasets import DataLoader, FoldersDataset#, RecursiveFoldersDataset

    opt_gp = torch.optim.Adam(model.net_d.parameters(), lr=1e-3)

    print(' - Using Dataset: CelebA')
    cats = ['/data/celeba/']
    dset = FoldersDataset(cats, crop='center', img_size=args.resolution)

    loader = DataLoader(dset, batch_size=args.batchSize, num_workers=args.numWorkers, shuffle=False)
    diter = iter(loader)

    for epoch in range(5000):
        for _ in range(9999999):
            losses = {}
            b = args.batchSize
            model.opt_d.zero_grad(); model.opt_g.zero_grad()

            # Update D
            model.net_d.train()
            model.net_g.train()
            for di in range(args.numUpdatesD):
                realImgs = get_real_imgs(diter)
                b,c,h,w = realImgs.size()

                #if ii>25 and di > 0: model.net_d.eval()
                #if ii > 5 and np.random.randint(2) == 0: model.net_d.eval()
                #else: model.net_d.train()
                #if ii > 25 and np.random.randint(2) == 0: model.net_g.eval()
                #else: model.net_g.train()

                with torch.no_grad():
                    fakeImgs = model.forward_g(batchSize=b)

                #overridePenalty = args.penaltyFactor if di == args.numUpdatesD - 1 else 0
                #overridePenalty = None
                overridePenalty = 0
                loss_d,losses_d = model.loss_d(realImgs, fakeImgs, penaltyFactor=overridePenalty)
                loss_d.backward()

                model.opt_d.step()
                model.opt_d.zero_grad()

                if model.loss == 'wgan' and model.penaltyFactor == 0:
                    pass
                    #print(' - Clipping!')
                    #model.net_d.apply(weight_clip_func)

                for k,v in losses_d.items():
                    if k not in losses: losses[k] = v
                    else: losses[k] += v
                del loss_d, losses_d

            #model.net_d.eval()
            loss_ = model.loss_gp(realImgs, fakeImgs) * args.penaltyFactor
            #loss_ = model.loss_dragan(realImgs) * args.penaltyFactor
            loss_.backward()
            losses['gp'] = loss_.item()
            opt_gp.step()
            opt_gp.zero_grad()
            model.opt_g.zero_grad()

            # Update G
            model.opt_g.zero_grad()
            model.net_g.train()
            model.net_d.train()
            #model.net_d.eval()
            for gi in range(args.numUpdatesG):
                fakeImgs2 = model.forward_g(batchSize=b)

                loss_g,losses_g = model.loss_g(fakeImgs2)
                #loss_g,losses_g = model.loss_g(torch.cat((get_real_imgs(diter),fakeImgs2),0),secondHalf=True)
                loss_g.backward()

                model.opt_g.step()
                model.opt_g.zero_grad()
                model.opt_d.zero_grad() # Don't forget this

                for k,v in losses_g.items():
                    if k not in losses: losses[k] = v
                    else: losses[k] += v
            del loss_g, fakeImgs2, losses_g

            for k,v in losses.items(): losses_[k] = losses_.get(k,0) + v

            if ii % printEvery == 0:
                print(' -', ii,'|',end='')
                cut = 100 if model.loss == 'wgan' else 5
                if ii>0:
                    for k,v in losses_.items():
                        #print(' - {:<14s}: {}:'.format(k,v))
                        #print(' ({:<7}: {:<7})'.format(k,str(v/printEvery)[:7]), sep='',end='')
                        print(' ({:<7}: {:4.3g})'.format(k,(v/printEvery)), sep='',end='')
                        if ii >= printEvery*3:
                            if k in loss_history: loss_history[k].append(min(max(v/printEvery,-cut),cut))
                            else: loss_history[k] = [min(max(v/printEvery,-cut),cut)]
                            plt.plot(loss_history[k], label=k)
                        losses_[k] = 0
                print(' ')
                if ii >= printEvery*3:
                    plt.legend()
                    #plt.gca().set_ylim(-5,5) # Note: I set the y-limit here.
                    plt.savefig('out/gans/loss.jpg',dpi=200)
                    plt.clf()
            if ii % vizEvery == 0:
                with torch.no_grad():
                    model.eval()
                    #model.net_g.eval()
                    #model.net_d.eval()
                    realImgs = realImgs[:4]
                    fakeImgs = model.forward_g(batchSize=realImgs.size(0))
                    if isinstance(fakeImgs, list):
                        viz_msg_batch(model, realImgs, fakeImgs, ii)
                    else:
                        viz_batch(model, realImgs, fakeImgs, ii)
            if ii % saveEvery == 0 and ii > 5:
                print(' - saving.')
                model.save(epoch=ii, dir='saves/gans')
            ii = ii + 1
            del realImgs




if __name__ == '__main__':
    main()

main()
