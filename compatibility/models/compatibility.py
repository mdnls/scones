import torch
import torch.nn as nn
import numpy as np

def get_cost(transport):
    if(transport == "l2-sq"):
        c = lambda x, y: torch.sum((x-y)**2, dim=(1, 2, 3))[:, None]
    elif(transport == "mean-l2-sq"):
        c = lambda x, y: torch.mean((x - y) ** 2, dim=(1, 2, 3))[:, None]
    else:
        raise ValueError(f"{transport} is not a valid choice of transport metric.")
    return c

class Compatibility(nn.Module):
    def __init__(self, inp_density_param, outp_density_param, config):
        super(Compatibility, self).__init__()
        self.config = config
        self.transport_cost = get_cost(config.transport.cost)

        # based on regularization: need a method to output a density
        # and a method to output a regularization value, to be used in a dual objective
        r = config.transport.coeff
        if(config.transport.regularization == "entropy"):
            self.penalty_fn = lambda x, y: r * torch.exp((1/r)*self._violation(x, y) - 1)
            self.compatibility_fn = lambda x, y: torch.exp((1/r)*self._violation(x, y) - 1)
        elif(config.transport.regularization == "l2"):
            self.penalty_fn = lambda x, y: (1/(4*r)) * torch.relu(self._violation(x, y))**2
            self.compatibility_fn = lambda x, y: (1/(2*r)) * torch.relu(self._violation(x, y))
        else:
            raise ValueError("Invalid choice of regularization")

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

        if(reg_type == "entropy"):
            target_p_grad = torch.cat(torch.autograd.grad(outputs=list(self.outp_density_param(y)), inputs=[y]), dim=1)
            if (self.config.transport.cost == "l2-sq"):
                scale = 1
            elif (self.config.transport.cost == "mean-l2-sq"):
                scale = 1 / np.prod(y.shape[1:])
            else:
                scale = 1
            temp = 1 / self.config.transport.coeff
            transport_grad = temp * (target_p_grad + scale * (x - y))
            return transport_grad
        elif(reg_type == "l2"):
            soft_compatibility_fn = lambda x, y: (1/(2*self.coeff)) * nn.functional.softplus(self._violation(x, y))
            return torch.cat(torch.autograd.grad(outputs=list(torch.log(soft_compatibility_fn(x, y))), inputs=[y]), dim=1)