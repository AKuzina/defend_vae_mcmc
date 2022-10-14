import torch
from torchvision import datasets
from torch.utils.data import random_split
from PIL import Image
import os

from datasets.mnists import MNIST


class color_mnist(datasets.MNIST):
    def __init__(self, root, train=True, transform=None):
        super(color_mnist, self).__init__(root, train=train, transform=transform)
        self.colorify()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def colorify(self):
        """
        MNIST dataset as described in "ADVERSARIALLY ROBUST REPRESENTATIONS WITH SMOOTH ENCODERS":
        The ColorMNIST is constructed from the MNIST dataset by coloring each digit
        artificially with all of the colors corresponding to the seven of the eight corners
        of the RGB cube (excluding black).
        """

        x = self.data.unsqueeze(1).repeat(1, 3*7, 1, 1)
        idx = torch.cat([torch.LongTensor([i, j, k]) for i in range(2) for j in range(2)
                         for k in range(2)]).bool()[3:]
        x[:, ~idx] = 0.
        color_label = torch.LongTensor(range(7)).repeat(x.shape[0])
        class_label = self.targets.unsqueeze(1).repeat(1, 7).reshape(1, -1).squeeze(0)
        self.data = x.reshape(-1, 3, 28, 28)
        self.targets = torch.stack([class_label, color_label], 1)

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')


class ColorMNIST(MNIST):
    def __init__(self, args, binarize=True):
        super().__init__(args, binarize=binarize)

    def prepare_data(self):
        datasets.MNIST(self.root, train=True, download=True)
        datasets.MNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = color_mnist(self.root, train=True, transform=self.transforms)
            self.train, self.val = random_split(mnist_full, [55000*7, 5000*7])

        if stage == 'test' or stage is None:
            self.test = color_mnist(self.root, train=False, transform=self.transforms)

