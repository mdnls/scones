import numpy as np
import torch.nn as nn
import torch

class GaussianScore(nn.Module):
    def __init__(self, config):
        super(GaussianScore, self).__init__()
        self.mean = torch.nn.Parameter(torch.FloatTensor(config.data.mean).view((1, -1, 1)))
        self.precision = torch.nn.Parameter(torch.FloatTensor(np.linalg.inv(config.data.cov)))
        self.config = config

    def forward(self, x):
        dim = self.config.data.dim
        r = - self.precision @ (x.view((-1, dim, 1)) - self.mean)
        return r.view((-1, dim, 1, 1))
