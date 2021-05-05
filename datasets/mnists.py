import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets

import pytorch_lightning as pl
# fix mnist download problem (https://github.com/pytorch/vision/issues/1938)
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


class Binarize(object):
    def __call__(self, x):
        return torch.bernoulli(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MNIST(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.root = './datasets'

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            Binarize()
        ])
        self.dims = (1, 28, 28)
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size

    def prepare_data(self):
        datasets.MNIST(self.root, train=True, download=True)
        datasets.MNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.root, train=True, transform=self.transforms)
            self.train, self.val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.test = datasets.MNIST(self.root, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=6, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, self.test_batch_size, num_workers=6, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, self.test_batch_size, num_workers=6, shuffle=False)


class FashionMNIST(MNIST):
    def __init__(self, args):
        super().__init__(args)

    def prepare_data(self):
        datasets.FashionMNIST(self.root, train=True, download=True)
        datasets.FashionMNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.FashionMNIST(self.root, train=True,
                                               transform=self.transforms)
            self.train, self.val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.test = datasets.FashionMNIST(self.root, train=False,
                                              transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=6, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, self.test_batch_size, num_workers=6, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, self.test_batch_size, num_workers=6, shuffle=False)



