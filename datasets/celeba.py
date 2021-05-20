import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets

import pytorch_lightning as pl


class CelebA(pl.LightningDataModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.root = './datasets'
        self.transforms = transforms.Compose([
            transforms.Resize(45),
            transforms.CenterCrop((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.dims = (3, 32, 32)
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.tasks = ['Bald', 'Eyeglasses',  'Mustache', 'Smiling',
                      'Wearing_Lipstick', 'Wearing_Necklace']
        self.task_ids = [4, 15, 22, 31, 36, 37]

    def prepare_data(self):
        datasets.CelebA(self.root, split='all', download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train = datasets.CelebA(root=self.root, split='train',
                                         target_type='attr',
                                         transform=self.transforms)
            self.val = datasets.CelebA(root=self.root, split='valid',
                                       target_type='attr',
                                       transform=self.transforms)
            # keep only relevant labels
            self.train.attr = self.train.attr[:, self.task_ids]
            self.val.attr = self.val.attr[:, self.task_ids]
        if stage == 'test' or stage is None:
            self.test = datasets.CelebA(root=self.root, split='test',
                                        target_type='attr',
                                        transform=self.transforms)
            # keep only relevant labels
            self.test.attr = self.test.attr[:, self.task_ids]

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=6, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, self.test_batch_size, num_workers=6, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, self.test_batch_size, num_workers=6, shuffle=False)

