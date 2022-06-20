import torch
import torch.utils.data as data
from scipy.stats import multivariate_normal
import numpy as np
import scipy.linalg


# TODO: rewrite this janky implementation.
class Gaussian(data.Dataset):
    _repr_indent = 4
    def __init__(self, mean, cov, device, shape=None, iid_unit=False):
        self.device = device
        self.mean = torch.FloatTensor(mean).to(device)
        self.cov = torch.FloatTensor(cov).to(device)
        self.iid_unit = iid_unit
        self.dist = multivariate_normal(mean=mean, cov=cov, seed=2039)

        if(iid_unit and shape is None):
            self.shape = (1,)
        elif(shape is None):
            self.shape = mean.shape
        else:
            self.shape = shape

    def __getitem__(self, index):
        return torch.FloatTensor(self.dist.rvs().astype(np.float)).to(self.device), 0

    def __len__(self):
        return int(1e+4)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""
