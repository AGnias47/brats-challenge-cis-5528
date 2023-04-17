from torch import optim
from monai.networks.nets import SegResNet as MonaiSegResNet
from .nnet import NNet


class SegResNet(NNet):
    def __init__(self):
        model = MonaiSegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
        )
        optimizer = optim.Adam
        alpha = 0.0065
        super().__init__("segresnet", model, optimizer, alpha)
