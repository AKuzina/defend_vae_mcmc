from .mnists import MNIST, FashionMNIST
from .celeba import CelebA


def load_dataset(args, binarize=True):
    data_module = {
        'mnist':  MNIST,
        'fashion_mnist': FashionMNIST,
        'celeba': CelebA,
    }[args.dataset_name](args, binarize=binarize)

    img_size = {
        'mnist': [1, 28, 28],
        'fashion_mnist': [1, 28, 28],
        'celeba': [3, 32, 32],
    }[args.dataset_name]
    with args.unlocked():
        args.image_size = img_size
    return data_module, args
