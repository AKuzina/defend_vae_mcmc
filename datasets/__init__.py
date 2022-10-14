from .mnists import MNIST, FashionMNIST
from .color_mnist import ColorMNIST


def load_dataset(args, binarize=True):
    data_module = {
        'mnist':  MNIST,
        'fashion_mnist': FashionMNIST,
        'color_mnist': ColorMNIST,
    }[args.dataset_name](args, binarize=binarize)

    img_size = {
        'mnist': [1, 28, 28],
        'fashion_mnist': [1, 28, 28],
        'color_mnist': [3, 28, 28],
    }[args.dataset_name]
    # for classification in latent space
    n_classes = {
        'mnist': [10],
        'fashion_mnist': [10],
        'color_mnist': [10, 7],
    }[args.dataset_name]

    with args.unlocked():
        args.image_size = img_size
        args.n_classes = n_classes

    return data_module, args
