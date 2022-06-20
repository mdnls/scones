import torch
import torch.nn as nn
import numpy as np
from compatibility.models.special_metrics import FacenetMetric, LowresMetric, SSIMMetric

def _at(A, x):
    return torch.matmul(A, x[:, :, None])[:, :, 0]

def _dot(xT, Ax):
    return torch.sum(xT * Ax, dim=-1)

def get_cost(config):
    cost = config.transport.cost
    if(cost == "l2-sq"):
        c = lambda x, y: torch.sum((x.flatten(start_dim=1) - y.flatten(start_dim=1))**2, dim=1)[:, None]
    elif(cost == "mean-l2-sq"):
        c = lambda x, y: torch.mean((x.flatten(start_dim=1) - y.flatten(start_dim=1)) ** 2, dim=1)[:, None]
    elif(cost == "facenet"):
        c = FacenetMetric(config)
    elif(cost == "lowres-8px"):
        c = LowresMetric(lowres=8, config=config) # lowres = 8px
    elif(cost == "ssim"):
        c = SSIMMetric(config=config)
    else:
        raise ValueError(f"{cost} is not a valid choice of cost metric.")
    return c

class Compatibility(nn.Module):
    def __init__(self, inp_density_param, outp_density_param, config, swap_xy=False):
        super(Compatibility, self).__init__()
        self.config = config
        self.transport_cost = get_cost(config)

        # based on regularization: need a method to output a density
        # and a method to output a regularization value, to be used in a dual objective
        r = config.transport.coeff
        if(config.transport.regularization == "entropy"):
            self.penalty_fn = lambda x, y: r * torch.exp((1/r)*self._violation(x, y) - 1)
            self.compatibility_fn = lambda x, y: torch.exp((1/r)*self._violation(x, y) - 1)
        elif (config.transport.regularization == "cuturi-entropy"):
            self.penalty_fn = lambda x, y: r*(torch.exp((1 / r) * self._violation(x, y))-1)
            self.compatibility_fn = lambda x, y: torch.exp((1 / r) * self._violation(x, y))
        elif(config.transport.regularization == "l2"):
            self.penalty_fn = lambda x, y: (1/(4*r)) * torch.relu(self._violation(x, y))**2
            self.compatibility_fn = lambda x, y: (1/(2*r)) * torch.relu(self._violation(x, y))
        else:
            raise ValueError("Invalid choice of regularization")

        if(swap_xy):
            self.inp_density_param_net = outp_density_param
            self.outp_density_param_net = inp_density_param
        else:
            self.inp_density_param_net = inp_density_param
            self.outp_density_param_net = outp_density_param

    def _violation(self, x, y):
        if(type(x) == tuple and type(y) == tuple):
            t_cost = sum([self.transport_cost(ex, why) for ex, why in zip(x, y)])
        else:
            t_cost = self.transport_cost(x, y)
        return self.inp_density_param_net(x) + self.outp_density_param_net(y) - t_cost

    def penalty(self, x, y):
        return self.penalty_fn(x, y)

    def forward(self, x, y):
        return self.compatibility_fn(x, y)

    def inp_density_param(self, x, *args):
        return self.inp_density_param_net(x, *args)

    def outp_density_param(self, y, *args):
        return self.outp_density_param_net(y, *args)

    def score(self, x, y):
        # For H(x, y) a compatibility function, this method computes grad_y log H(x, y) for a fixed x and y.
        reg_type = self.config.transport.regularization
        temp = 1 / self.config.transport.coeff
        cost = self.config.transport.cost
        if (reg_type == "entropy" and cost in ["l2-sq", "mean-l2-sq"]):
            target_p_grad = torch.cat(torch.autograd.grad(outputs=list(self.outp_density_param(y)), inputs=[y]), dim=1)
            scale = 2 if (cost == "l2-sq") else 2 / np.prod(y.shape[1:])
            transport_grad = temp * (target_p_grad + scale * (x - y))
        elif(reg_type == "entropy" or reg_type == "cuturi-entropy"):
            cpat_fn = lambda x, y: temp * self._violation(x, y)
            transport_grad = torch.cat(torch.autograd.grad(outputs=list(cpat_fn(x, y)), inputs=[y]), dim=1)
        elif(reg_type == "l2"):
            if(hasattr(self.config.model, "beta")):
                beta = self.config.model.beta
            else:
                beta = 1 # default pytorch value
            soft_cpat_fn = lambda x, y: (1/2) * temp * nn.functional.softplus(self._violation(x, y), beta=beta)
            transport_grad = torch.cat(torch.autograd.grad(outputs=list(torch.log(soft_cpat_fn(x, y))), inputs=[y]), dim=1)

        return transport_grad

    def density(self, x, y):
        reg_type = self.config.transport.regularization
        temp = 1 / self.config.transport.coeff
        if (reg_type == "entropy"):
            return torch.exp(temp * self._violation(x, y) - 1)
        elif(reg_type == "cuturi-entropy"):
            return torch.exp(temp * self._violation(x, y))
        elif(reg_type == "l2"):
            raise NotImplementedError()
