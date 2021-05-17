from facenet_pytorch import InceptionResnetV1
import pytorch_msssim
import torch.nn as nn
import torch
from datasets import inverse_data_transform

class LearnedMetric(nn.Module):
    def __init__(self, repr_source, repr_target, repr_cost):
        '''
        Implements a learned metric to compare data, of the form d(x, y) = c(Rs(x), Rt(y)) where Rs, Rt map x, y respectively
            to some representation vector and where c is a l2, l2-squared, or other cost.
        Args:
            repr_source: the function Rs which maps source distribution data to representation vectors
            repr_target: the function Rs which maps source distribution data to representation vectors
            repr_cost: choice of c. Either 'l2-sq' or 'mean-l2-sq'.
        '''
        super(LearnedMetric, self).__init__()

        assert repr_cost in ['l2-sq', 'mean-l2-sq'], f"{repr_cost} is an invalid choice of repr_cost."
        self.repr_source = repr_source
        self.repr_target = repr_target

        if (repr_cost == "l2-sq"):
            self.repr_cost = lambda x, y: torch.sum( (x.flatten(start_dim=1) - y.flatten(start_dim=1))**2, dim=1)[:, None]
        elif (repr_cost == "mean-l2-sq"):
            self.repr_cost = lambda x, y: torch.mean( (x.flatten(start_dim=1) - y.flatten(start_dim=1))**2 , dim=1)[:, None]

    def forward(self, x, y):
        return self.repr_cost(self.repr_source(x), self.repr_target(y))


class FacenetMetric(nn.Module):
    def __init__(self, config):
        '''
        A learned metric between distances which computes a Euclidean distance between a learned representation for each
            image.

        Args:
            repr_cost: Euclidean cost to use between learned image representations. One of ['l2-sq', 'mean-l2-sq'].
        '''
        super(FacenetMetric, self).__init__()
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()
        self.resize = nn.Upsample(size=160, mode="bilinear", align_corners=True) # 160px is default image size
        self.metric = LearnedMetric(repr_source=self._repr_source, repr_target=self._repr_target, repr_cost='mean-l2-sq')
        self.config = config

    def preprocess(self, img_batch, is_source):
        if(is_source):
            resized_batch = inverse_data_transform(self.config.source, self.resize(img_batch))
        else:
            resized_batch = inverse_data_transform(self.config.target, self.resize(img_batch))
        return 2 * resized_batch - 1

    def _repr_source(self, img_batch):
        return self.resnet(self.preprocess(img_batch, is_source=True))

    def _repr_target(self, img_batch):
        return self.resnet(self.preprocess(img_batch, is_source=False))

    def forward(self, x, y):
        return self.metric(x, y)


class LowresMetric(nn.Module):
    def __init__(self, lowres, config):
        super(LowresMetric, self).__init__()
        # Upsample will actually resize inputs to arbitrary sizes. It is used here for downsampling.
        self.downsampler = nn.Upsample(size=lowres, mode='bilinear', align_corners=True)
        self.config = config
        self.metric = LearnedMetric(repr_source=self.downsampler,
                                    repr_target=self.downsampler,
                                    repr_cost='mean-l2-sq')

    def forward(self, x, y):
        return self.metric(x, y)


class SSIMMetric(nn.Module):
    def __init__(self, config):
        super(SSIMMetric, self).__init__()
        self.config = config

    def forward(self, x, y):
        return pytorch_msssim.ssim(x, y)
