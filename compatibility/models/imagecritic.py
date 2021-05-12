import torch
import torch.nn as nn
from .simple import ReLU_MLP, ReLU_CNN

class FCImageCritic(nn.Module):
    def __init__(self, input_im_size, input_channels, hidden_layer_dims):
        super(FCImageCritic, self).__init__()
        self.input_W = input_im_size
        self.input_C = input_channels
        self.hidden_layer_dims = hidden_layer_dims
        self.inp_projector = nn.Linear(in_features=self.input_W ** 2 * self.input_C,
                                       out_features=hidden_layer_dims[0])
        self.outp_projector = nn.Linear(in_features = hidden_layer_dims[-1], out_features=1)
        self.hidden = ReLU_MLP(layer_dims=hidden_layer_dims)
    def forward(self, inp_image):
        return self.outp_projector(nn.functional.relu(self.hidden(self.inp_projector(inp_image.flatten(start_dim=1)))))

class ImageCritic(nn.Module):
    def __init__(self, input_im_size, layers, channels, batchnorm=True):
        '''
        The critic accepts an image or batch of images and outputs a scalar or batch of scalars.
        Given the input image size and a desired number of layers, the input image is downsampled at each layer until
            is a set of 4x4 feature maps. The scalar output is a regression over the channel dimension.

        Arguments:
            - input_im_size (int): size of input images, which are assumed to be square
            - layers (int): number of layers
            - channels (list of int): number of input channels to each network layer, including the first layer.
            - batchnorm: if True, use batchnorm in hidden layers
        '''
        super(ImageCritic, self).__init__()


        scale = (input_im_size/4)**(-1/ (layers-1) )

        imdims = [input_im_size] + \
                 [int(input_im_size * (scale**k)) for k in range(1, layers-1)]  + \
                 [8]

        imdims = [(x, x) for x in imdims]

        self.net = ReLU_CNN(imdims, channels, filter_size=3, output="none", batchnorm=batchnorm)
        self.linear = torch.nn.Linear(8 * 8 * channels[-1], 1)

    def forward(self, image, *args):
        # the *args is a hack to allow extra arguments, for ex, if one wants to pass two input images.
        # TODO: nicely implement functionality for networks which take in two or more arguments.
        if(type(image) == tuple):
            args = image[1:]
            image = image[0]
        if(len(args) > 0):
            image = torch.cat([image] + list(args), dim=1)
        return self.linear(torch.flatten(self.net(image), start_dim=1))
    def clip_weights(self, c):
        for p in self.parameters():
            p.data.clamp_(-c, c)
