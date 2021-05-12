import torch
import torch.nn as nn

class ReLU_MLP(nn.Module):
    def __init__(self, layer_dims, output="linear", layernorm=False):
        '''
        A generic ReLU MLP network.

        Arguments:
            - layer_dims: a list [d1, d2, ..., dn] where d1 is the dimension of input vectors and d1, ..., dn
                        is the dimension of outputs of each of the intermediate layers.
            - output: output activation function, either "sigmoid" or "linear".
            - layernorm: if True, apply layer normalization to the input of each layer.
        '''
        super(ReLU_MLP, self).__init__()
        layers = []
        for i in range(1, len(layer_dims) - 1):
            if (layernorm and i != 1):
                layers.append(nn.LayerNorm(layer_dims[i - 1]))
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            layers.append(nn.ReLU(layer_dims[i - 1]))
        if (output == "sigmoid"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
            layers.append(nn.Sigmoid())
        if (output == "linear"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        self.layers = layers
        self.out = nn.Sequential(*layers)

    def forward(self, inp, *args):
        if (type(inp) == tuple):
            args = inp[1:]
            inp = inp[0]
        if (len(args) > 0):
            inp = torch.cat([inp] + list(args), dim=1)
        return self.out(inp)

    def clip_weights(self, c):
        for layer in self.layers:
            if (isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm)):
                layer.weight.data = torch.clamp(layer.weight.data, -c, c)
                layer.bias.data = torch.clamp(layer.bias.data, -c, c)

class ReLU_CNN(nn.Module):
    def __init__(self, imdims, channels, filter_size, output="sigmoid", batchnorm=False):
        '''
        A generic ReLU CNN network.

        Arguments:
            - imdims: a length-2 tuple of integers, or a list of these tuples. Each is image dimensions in HW format.
                If input is a list, the list must have one fewer item than the length of channels. The output of each
                layer is resized to the given dimensions.
            - channels: a list [c1, c2, ..., cn] where c1 is the number of input channels and c2, ..., cn
                    is the number of output channels of each intermediate layer.

                    The final layer does not resize the image so len(channels) = len(imdims) + 1 is required.
            - filter_size: size of convolutional filters in each layer.
            - output: output activation function, either "sigmoid" or "tanh".
            - batchnorm: if True, apply batch normalization to the input of each layer.
        '''

        super(ReLU_CNN, self).__init__()
        layers = []

        assert all([type(x) == int for x in channels]), "Channels must be a list of integers"
        def istwotuple(x):
            return (type(x) == tuple) and (len(x) == 2) and (type(x[0]) == int) and (type(x[1]) == int)

        if(istwotuple(imdims)):
            imdims = [imdims for _ in range(len(channels) + 1)]
        elif(all([istwotuple(x) for x in imdims])):
            assert len(imdims)+1 == len(channels), "The length of channels must be one greater than the length of imdims."
        else:
            raise ValueError("Input image dimensions are not correctly formatted.")

        self.imdims = imdims

        padding = int((filter_size - 1) / 2)
        for i in range(1, len(channels) - 1):
            if (batchnorm and not i == 1):
                layers.append(nn.BatchNorm2d(num_features=channels[i-1]))
            layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=filter_size, padding=padding))
            if(imdims[i-1] != imdims[i]):
                layers.append(torch.nn.Upsample(imdims[i], mode='bilinear', align_corners=True))
            layers.append(nn.ReLU(channels[i - 1]))
        if (output == "sigmoid"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
            layers.append(nn.Sigmoid())
        elif (output == "tanh"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
            layers.append(nn.Tanh())
        elif (output == "none"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
        else:
            raise ValueError("Unrecognized output function.")
        self.layers = layers
        self.out = nn.Sequential(*layers)

    def forward(self, inp):
        return self.out(inp)

    def clip_weights(self, c):
        for layer in self.layers:
            if(isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d)):
                layer.weight.data = torch.clamp(layer.weight.data, -c, c)
                layer.bias.data = torch.clamp(layer.bias.data, -c, c)

