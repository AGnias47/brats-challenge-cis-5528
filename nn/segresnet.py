from torch import optim
from monai.networks.nets import SegResNet as MonaiSegResNet
from .nnet import NNet


class SegResNet(NNet):
    def __init__(self, input_channels=4, output_channels=3, learning_rate=0.15):
        model = MonaiSegResNet(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=output_channels,
        )
        optimizer = optim.Adam
        super().__init__("segresnet", model, optimizer, learning_rate)
