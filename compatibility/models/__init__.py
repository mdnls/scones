import torch
from .compatibility import *
from .imagecritic import *
from .simple import *

def get_compatibility(config):
    source_shape = (config.source.data.channels,
                    config.source.data.image_size,
                    config.source.data.image_size)

    target_shape = (config.target.data.channels,
                    config.target.data.image_size,
                    config.target.data.image_size)

    if(config.model.architecture.upper() == "FCN"):
        source_critic = FCImageCritic(input_im_size = source_shape[1],
                                      input_channels = source_shape[0],
                                      hidden_layer_dims = config.model.hidden_layers).to(config.device)

        target_critic  =  FCImageCritic(input_im_size = target_shape[1],
                                      input_channels = target_shape[0],
                                      hidden_layer_dims = config.model.hidden_layers).to(config.device)

    elif(config.model.architecture.upper() == "CNN"):
        source_critic = ImageCritic(input_im_size = source_shape[1],
                                    channels = [source_shape[0]] + config.model.hidden_layers,
                                    layers = len(config.model.hidden_layers),
                                    batchnorm = False).to(config.device)

        target_critic = ImageCritic(input_im_size = target_shape[1],
                                    channels = [target_shape[0]] + config.model.hidden_layers,
                                    layers = len(config.model.hidden_layers),
                                    batchnorm = False).to(config.device)
    else:
        raise ValueError(f"{config.model.architecture} is not a valid choice of architecture.")

    cpat = Compatibility(inp_density_param=source_critic,
                         outp_density_param=target_critic,
                         config=config).to(config.device)

    return cpat

