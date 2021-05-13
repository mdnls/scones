import torch
import torch.utils.data as data
from scipy.stats import multivariate_normal
import numpy as np

class Gaussian(data.Dataset):
    _repr_indent = 4
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.dim = len(mean)
        self.dist = multivariate_normal(mean=self.mean, cov=self.cov, allow_singular=True, seed=2039)

    def __getitem__(self, index):
        # Return in the format of a 1x1 px image.
        return self.dist.rvs(size=(1,)).reshape((-1, 1, 1)).astype(np.float), 0

    def __len__(self):
        return int(1e+6)

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
