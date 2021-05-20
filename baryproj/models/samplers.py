import torch
import torch.nn as nn
import math
from compatibility.models import ReLU_CNN, ReLU_MLP
from ncsn.models.normalization import get_normalization
from ncsn.models.layers import get_act, ResidualBlock, RefineBlock
from datasets import data_transform

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
    elif (config.model.architecture == "res"):
        return ResidualSampler(config).to(config.device)
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



class ResidualSampler(nn.Module):
    def __init__(self, config):
        super(ResidualSampler, self).__init__()
        self.source_logit_transform = config.source.data.logit_transform
        self.source_rescaled = config.source.data.rescaled
        self.norm = get_normalization(config, conditional=False)

        self.ngf = ngf = config.model.ngf
        self.act = act = get_act(config)
        self.config = config
        self.begin_conv = nn.Conv2d(config.source.data.channels, ngf, 3, stride=1, padding=1)

        self.normalizer = self.norm(ngf)
        self.end_conv = nn.Conv2d(ngf, config.source.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        if config.source.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=True, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=False, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x):
        if not self.source_logit_transform and not self.source_rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)
        output = torch.sigmoid(output)
        return data_transform(self.config.target, output)


