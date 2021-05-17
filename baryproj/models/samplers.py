import torch
import torch.nn as nn
import math
from compatibility.models import ReLU_CNN, ReLU_MLP

# so, do you like jazz?
def get_bary(config):
    if(config.model.architecture == "conv"):
        return ImageSampler(
            input_im_size=config.source.data.image_size,
            output_im_size=config.target.data.image_size,
            upsampler_layers=1+len(config.model.upsampler_hidden_channels),
            downsampler_layers=1+len(config.model.downsampler_hidden_channels),
            upsampler_channels=[config.model.bottleneck_dim] +
                               config.model.upsampler_hidden_channels +
                               [config.target.data.channels],
            downsampler_channels=[config.source.data.channels] +
                                 config.model.downsampler_hidden_channels +
                                 [config.model.bottleneck_dim]).to(config.device)
    elif(config.model.architecture == "fcn"):
        return FCSampler(
            input_im_size=config.source.data.image_size,
            input_channels=config.source.data.channels,
            output_im_size=config.target.data.image_size,
            output_channels=config.target.data.channels,
            hidden_layer_dims=config.model.hidden_layer_dims
        ).to(config.device)
    else:
        raise ValueError(f"{config.model.architecture} is not a recognized architecture.")


class ImageSampler(nn.Module):
    def __init__(self, input_im_size, output_im_size, downsampler_layers, upsampler_layers, downsampler_channels, upsampler_channels):
        '''
        Bottleneck architecture that takes an input image from the source distribution, maps it to an intermediate
            latent code through a downsampling network, which is then mapped through an upsampling network to a
            target code.

        Arguments:
            - input_im_size (int): side length of input images, which are assumed square
            - output_im_size (int): side length of output images, which are assumed square
            - downsampler_layers (int): total number of downsampling layers. Each downsamples by a constant factor,
                reducing side length from input_im_size to 8.
            - upsampler_layers (int): total number of upsampling layers. Each upsamples by a constant factor,
                increasing side length from 8 to output_im_size.
            - downsampler_channels (int): number of channels in each downsampling layer
            - upsampler_channels (int): number of channels in each upsampling layer
        '''
        super(ImageSampler, self).__init__()

        # Compute the scale so that (downsampler_layers - 1) downsamples by constant scale will scale from initial image width to 8x8.
        # the offset -1 for layer counts is because the final layer does not include an upsampling operation
        # which is to increase output fidelity by avoiding upsampling blur in the final output.
        downsampler_scale = (input_im_size/8)**(-1/ (downsampler_layers-1) )
        upsampler_scale = (output_im_size/8)**(1/(upsampler_layers-1) )

        downsampler_imdims = [input_im_size] + \
                             [math.ceil(input_im_size * (downsampler_scale**k)) for k in range(1, downsampler_layers-1)] + \
                             [8]
        downsampler_imdims = [(x, x) for x in downsampler_imdims]
        upsampler_imdims = [8] + \
                             [math.ceil(8 * (upsampler_scale**k)) for k in range(1, upsampler_layers-1)] + \
                             [output_im_size]
        upsampler_imdims = [(x, x) for x in upsampler_imdims]

        self.downsampler_imdims = downsampler_imdims
        self.upsampler_imdims = upsampler_imdims
        self.downsampler_channels = downsampler_channels
        self.upsampler_channels = upsampler_channels
        self.downlinproj = nn.Linear(in_features=8*8*downsampler_channels[-1], out_features=downsampler_channels[-1])
        self.uplinproj = nn.Linear(in_features=upsampler_channels[0], out_features=8*8*upsampler_channels[0])
        self.downsampler = ReLU_CNN(imdims=downsampler_imdims, channels=downsampler_channels, filter_size=3, output="none", batchnorm=True)
        self.upsampler = ReLU_CNN(imdims=upsampler_imdims, channels=upsampler_channels, filter_size=3, output="sigmoid", batchnorm=True)

    def forward(self, input_image):
        down_spatial_features = self.downsampler(input_image)
        autoenc_hidden = self.downlinproj(down_spatial_features.flatten(start_dim=1))
        up_spatial_features = self.uplinproj(autoenc_hidden).reshape((-1, self.upsampler_channels[0], 8, 8))
        return self.upsampler(up_spatial_features)

class FCSampler(nn.Module):
    def __init__(self, input_im_size, input_channels, output_im_size, output_channels, hidden_layer_dims):
        super(FCSampler, self).__init__()
        self.input_W = input_im_size
        self.input_C = input_channels
        self.output_W = output_im_size
        self.output_C = output_channels
        self.hidden_layer_dims = hidden_layer_dims
        self.inp_projector = nn.Linear(in_features=self.input_W ** 2 * self.input_C,
                                       out_features=hidden_layer_dims[0])
        self.outp_projector = nn.Linear(in_features = hidden_layer_dims[-1],
                                        out_features=self.output_W **2 * self.output_C)
        self.hidden = ReLU_MLP(layer_dims=hidden_layer_dims, layernorm=False)

    def forward(self, inp_image):
        inp_img_scale = 2 * inp_image - 1
        x = self.outp_projector(nn.functional.relu(self.hidden(self.inp_projector(inp_img_scale.flatten(start_dim=1)))))
        return (1/2) * (torch.tanh(x).reshape((-1, self.output_C, self.output_W, self.output_W)) + 1)