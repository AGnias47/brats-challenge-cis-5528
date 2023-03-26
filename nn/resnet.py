from torch import optim
from monai.networks.nets import SegResNet
from .nnet import NNet


class ResNet(NNet):
    def __init__(self):
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
        )
        optimizer = optim.Adam
        alpha = 0.0065
        super().__init__("resnet", model, optimizer, alpha)
