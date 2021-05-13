import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN, MNIST, USPS
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.gaussian import Gaussian
from torch.utils.data import Subset
import numpy as np


def get_dataset(args, config):
    if config.data.dataset == 'CIFAR10':
        if(config.data.random_flip):
            dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True, download=True,
                              transform= transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()]))
            test_dataset = CIFAR10(os.path.join('datasets', 'cifar10_test'), train=False, download=True,
                                   transform=transforms.Compose([
                                    transforms.Resize(config.data.image_size),
                                    transforms.ToTensor()]))

        else:
            dataset = CIFAR10(os.path.join('datasets', 'cifar10'), train=True, download=True,
                              transform= transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()]))
            test_dataset = CIFAR10(os.path.join('datasets', 'cifar10_test'), train=False, download=True,
                                   transform=transforms.Compose([
                                transforms.Resize(config.data.image_size),
                                transforms.ToTensor()]))

    elif config.data.dataset == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)

    elif(config.data.dataset == "CELEBA-Pix"):
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(8),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(8),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join('datasets', 'celeba'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(8),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)

    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join('datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join('datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join('datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset == "MNIST":
        if config.data.random_flip:
            dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.ToTensor(),
                                               transforms.Resize(config.data.image_size)
                                           ]))
        else:
            dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Resize(config.data.image_size)
                                           ]))
        test_dataset = MNIST(root=os.path.join('datasets', 'MNIST'),
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Resize(config.data.image_size)
                                           ]))
    elif config.data.dataset == "USPS":
        if config.data.random_flip:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(20), # resize and pad like MNIST
                                               transforms.Pad(4),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.ToTensor(),
                                               transforms.Resize(config.data.image_size)
                                           ]))
        else:
            dataset = USPS(root=os.path.join('datasets', 'USPS'),
                                           train=True,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(20), # resize and pad like MNIST
                                               transforms.Pad(4),
                                               transforms.ToTensor(),
                                               transforms.Resize(config.data.image_size)
                                           ]))
        test_dataset = USPS(root=os.path.join('datasets', 'USPS'),
                                            train=False,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(20),  # resize and pad like MNIST
                                                transforms.Pad(4),
                                                transforms.ToTensor(),
                                                transforms.Resize(config.data.image_size)
                                            ]))
    elif(config.data.dataset.upper() == "GAUSSIAN"):
        if(config.data.isotropic):
            dim = config.data.dataset.dim
            rank = config.data.dataset.rank
            cov = np.diag( np.pad(np.ones((rank,)), [(0, dim - rank)]) )
            mean = np.zeros((dim,))
        else:
            cov = np.array(config.data.cov)
            mean = np.array(config.data.mean)
        dataset = Gaussian(cov=cov, mean=mean)
        test_dataset = Gaussian(cov=cov, mean=mean)
    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X.float()

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)