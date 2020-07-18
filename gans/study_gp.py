import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2
import visdom

'''
I'd like some intution on exactly *what* a gradient penalty does to a network.

There is a lot of theory about why to use it and such, but I am curious what it
does in terms of network weights and how it changes each layer's inputs/outputs.

Besides the theory in the gan literature, I think there is an intuitive explanation
that it strengthens the information flow directly from prediction to inputs.
This is the whole idea of the two-sidedness of the penalty, of course, but I think controlling
that information flow is more important then the lipschitz effect.
I think it prevents the network from learning spurious patterns and sticking to local optima,
and forces the input to have a far reaching signal rather than paying attention to such patterns.
'''

def normal_init(m):
    if hasattr(m,'weight'):
        m.weight.data.normal_(std=.02)

'''
How does a gradient penalty affect a randomly initialized two layer net?
'''
def run_gp():
    dev = torch.device('cuda')
    net = nn.Sequential(
            nn.Conv2d(16, 32, 3,1,1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3,1,1, bias=False)).to(dev)
    net.apply(normal_init)
    l = list(net.children())[0]
    iniWeightStd, iniWeightNorm = l.weight.std().item(), l.weight.norm().item()


    #opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)


    x0 = torch.randn(1,16,100,100, requires_grad=True, device=dev)
    y0 = net(x0)
    b,c1,h1,w1 = x0.size()
    _,c2,h2,w2 = y0.size()

    last_grad = None
    grad_corrs = []
    for i in range(100):
        x = torch.randn(1,16,100,100, requires_grad=True, device=dev)
        y = net(x)


        grads = torch.autograd.grad(
                outputs=y, inputs=[x],
                grad_outputs=torch.ones_like(y.data),
                retain_graph=True,only_inputs=True,create_graph=True)[0]

        print(' - Grads std: {}, norm {}'.format(grads.std().item(), grads.norm().item()))

        gp_target = 1
        gp = ((grads.view(b, -1) - gp_target)**2).mean()
        print(' - GP:', gp.item())
        # todo: Histogram of gps, over space

        gp.backward()

        if last_grad is not None:
            c = (F.normalize(l.weight.grad.view(1,-1)) * F.normalize(last_grad.view(1,-1))).sum().item()
            print( ' - grad corr:', c)
            grad_corrs.append(c)


        opt.step()
        last_grad = l.weight.grad.clone()
        opt.zero_grad()

    y2 = net(x0)

    # What did it do to the weight?
    print(' - Initial weight std: {}, norm {}'.format(iniWeightStd, iniWeightNorm))
    print(' - Final   weight std: {}, norm {}'.format(l.weight.std().item(),l.weight.norm().item()))

    # What did it do to the outputs?
    print(' - Final dy:', (y2 - y0).mean(-1).mean(-1)[:4,:4].cpu().detach().numpy(),'...')
    #print(' - Final dy spatial mean:', (y2 - y0).view(b,c2,h2*w2).mean(-1))
    #print(' - Final dy spatial std :', (y2 - y0).view(b,c2,h2*w2).std(-1))
    print(' - Final dy full    mean:', (y2 - y0).mean().item())
    print(' - Final dy full    std :', (y2 - y0).std().item())


    # How did the GP's gradient change over the course of optim?
    # Note: we are asking essentially about 2nd deriv.
    print(' - Avg One-Step GP grad correlation:', np.mean(grad_corrs))


def run_on(x, graph, node):
    pass


def run_grads():
    #x = torch.autograd.Variable(torch.FloatTensor([2]),True)
    #y = torch.autograd.Variable(torch.FloatTensor([3]),True)
    x = torch.arange(0,5., requires_grad=True)
    y = torch.arange(5,10., requires_grad=True)
    z = x*y

    grads = torch.autograd.grad(
            outputs=z, inputs=[x,y],
            grad_outputs=torch.ones_like(z.data),
            create_graph=True)

    print(' Gradient test:')
    print('   - x    :', x)
    print('   - y    :', y)
    print('   - z=x*y:', z)
    print('   - dx   :', grads[1])
    print('   - dy   :', grads[0])

    w = torch.arange(0,5*5., requires_grad=True).view(5,5,1)
    y = torch.arange(5,10., requires_grad=True).view(1,5)
    z = y @ w

    grads = torch.autograd.grad(
            outputs=z, inputs=[w,y],
            grad_outputs=torch.ones_like(z.data),
            create_graph=True)

    print(' Gradient test:')
    print('   - w    :\n', w.squeeze().data.numpy())
    print('   - y    :\n', y.squeeze().data.numpy())
    print('   - z=x*y:\n', z.squeeze().data.numpy())
    print('   - dy   :\n', grads[1].squeeze().data.numpy())
    print('   - dw   :\n', grads[0].squeeze().data.numpy())



'''
Looks like for a pretrained resnet while in train mode, solid color images
give crazy values for gradients, which could cause instability
early in training.

According to the 'which gan training methods converge' paper, it is okay
to use a GP with BNs. It must be that for WGAN-GP, the problem lies in
the wasserstein distance and not just the GP.
'''
def load_img(p):
    i = cv2.cvtColor(cv2.imread(p),cv2.COLOR_RGB2BGR)
    i = torch.from_numpy(cv2.resize(i, (256,256))).to(torch.float32)
    return i
def run_resnet(train=True):
    from torchvision.models import resnet50
    #from torchvision.ops.misc import FrozenBatchNorm2d
    #m = resnet50(True,norm_layer=FrozenBatchNorm2d).eval() # eval=train
    m = resnet50(True)
    conv1 = m.conv1
    m = nn.Sequential(*list(m.children())[:9])
    m = m.train(train)
    print('\n')
    print(' +++ ', 'Train' if train else 'Eval', 'Mode +++')

    imgs  = ['/home/slee/Downloads/field2.jpg']
    imgs += ['/home/slee/Downloads/field1.jpg']
    imgs += ['/home/slee/Downloads/canyon_hard1.jpg']
    imgs += ['/home/slee/Downloads/canyon_hard2.jpg']
    imgs += ['/data/wikiart/Rococo/antoine-pesne_jakob-von-keith.jpg']
    imgs += ['/data/wikiart/Rococo/allan-ramsay_robert-wood.jpg']
    imgs += ['/data/wikiart/Rococo/allan-ramsay_self-portrait.jpg']
    imgs += ['/data/wikiart/Rococo/allan-ramsay_queen-charlotte.jpg']
    imgs += ['/data/wikiart/Rococo/antoine-pesne_marianne-cochois.jpg']
    imgs += ['/data/wikiart/Rococo/antoine-pesne_portrait-of-frederick-ii.jpg']
    realImg = torch.stack([load_img(p) for p in imgs])
    realImg = realImg.div_(255.).permute(0,3,1,2).sub_(.5).div_(.4)
    realImg = torch.autograd.Variable(realImg,True)
    fakeImg = torch.randn_like(realImg, requires_grad=True) * 1


    def exec_graph(outReal, outFake):
        d_img_real,d_conv_real = torch.autograd.grad(
                outputs=outReal, inputs=[realImg,conv1.weight],
                grad_outputs=torch.ones_like(outReal),
                create_graph=True)
        d_img_fake,d_conv_fake = torch.autograd.grad(
                outputs=outFake, inputs=[fakeImg,conv1.weight],
                grad_outputs=torch.ones_like(outFake),
                create_graph=True)

        metric = torch.std
        metric = lambda t: t.norm().data
        #metric = lambda t: t.size()

        print(' dconv/dreal:', metric(d_conv_real))
        print(' dimg /dreal:', metric(d_img_real))
        print(' dconv/dfake:', metric(d_conv_fake))
        print(' dimg /dfake:', metric(d_img_fake))
        print(' gp real    :', metric(d_img_real.view(-1)))
        print(' gp fake    :', metric(d_img_fake.view(-1)))

    print('\n === Raw Output Vector ===')
    pred_real = m(realImg)
    pred_fake = m(fakeImg)
    exec_graph(pred_real, pred_fake)
    print('\n === Output Vector Norm ===')
    pred_real = m(realImg).norm()
    pred_fake = m(fakeImg).norm()
    exec_graph(pred_real, pred_fake)
    print('\n === Output Vector Distance To One ===')
    pred_real = ((m(realImg).norm() - 1) ** 2).mean()
    pred_fake = ((m(fakeImg).norm() - 1) ** 2).mean()
    exec_graph(pred_real, pred_fake)

#run_gp()
#run_grads()
run_resnet(False)
run_resnet(True)


