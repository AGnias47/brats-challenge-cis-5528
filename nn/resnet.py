from torch import optim
from monai.networks.nets import SegResNet
from .nnet import NNet


class ResNet(NNet):
    def __init__(self):
        self.name = "resnet"
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
        )
        optimizer = optim.Adam
        alpha = 1e-2
        super().__init__(model, optimizer, alpha)
