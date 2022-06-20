import torch.optim as optim

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        if(hasattr(config.optim, "beta2")):
            beta2 = config.optim.beta2
        else:
            beta2 = 0.999

        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, beta2), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == "LBFGS":
        return optim.LBFGS(parameters, lr=config.optim.lr)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
