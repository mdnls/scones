import numpy as np
import torch.nn as nn
import torch

class GaussianScore(nn.Module):
    def __init__(self, config):
        super(GaussianScore, self).__init__()
        mean = torch.FloatTensor(config.data.mean).to(config.device).view((1, -1, 1))
        cov_pinv = torch.FloatTensor(np.linalg.pinv(config.data.cov)).to(config.device)
        self.mean = torch.nn.Parameter(mean)
        self.precision = torch.nn.Parameter(cov_pinv)
        self.config = config

    def forward(self, x):
        dim = self.config.data.dim
        r = - self.precision @ (x.view((-1, dim, 1)) - self.mean)
        return r.view((-1, dim, 1, 1))
