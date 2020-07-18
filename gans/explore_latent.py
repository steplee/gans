import torch, torch.nn.functional as F, torch.nn as nn
import numpy as np, cv2
from gans.base import GanBase

with torch.no_grad():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default=None)
    args = parser.parse_args()

    assert(args.load)
    model, d = GanBase.load(args.load, restoreOpt=False)
    dev = torch.device('cuda')
    model.eval().to(dev)
    model.net_g.eval()

    # Lerp a -> b -> ...
    a = torch.randn(1, model.latentSize, device=dev)
    b = torch.randn(1, model.latentSize, device=dev)
    for i in range(1000):
        for t in np.linspace(0,1,100):
            z = a*(1-t) + b*t
            #z = F.normalize(z,dim=-1)
            y = model.forward_g(z)
            y = y.mul_(255).permute(0,2,3,1).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
            cv2.imshow('G(z)', cv2.cvtColor(y,cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        a = b
        b = torch.randn(1, model.latentSize, device=dev)
