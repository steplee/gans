import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision

from sys import stderr

from PIL import Image

from multiprocessing.dummy import Pool as ThreadPool
from torch.utils.data import IterableDataset, DataLoader

class FoldersDataset(IterableDataset):
    def __init__(self,
            roots,
            img_size=256,
            n=5000000,
            crop='center'):
        self.roots = roots

        self.cats = [d for d in roots if os.path.isdir(d)]
        self.files = []
        for cat in self.cats:
            for f in os.listdir(cat):
                if f.endswith('jpg') or f.endswith('.jpeg'):
                    self.files.append(os.path.join(cat,f))

        if crop == 'random':
            crop_ = [torchvision.transforms.RandomResizedCrop((img_size,img_size), scale=(.5,1), ratio=(1,1))]
        elif crop == 'center':
            crop_ = [torchvision.transforms.CenterCrop((img_size,img_size))]
        self.transform = torchvision.transforms.Compose((
                *crop_,
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()))

        print('\n - FoldersDataset has {} examples.\n'.format(len(self.files)))

    def __iter__(self):
        self.ii = 0
        return self

    def __next__(self):
        while True:
            try:
                file0 = random.choice(self.files)
                with open(file0, 'rb') as f:
                    img0 = Image.open(f).convert('RGB')
                img0 = self.transform(img0)
                return img0
            except:
                print(' - Failed one:', file0)
                continue

