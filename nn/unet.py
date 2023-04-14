from monai.networks.nets import UNet as MonaiUNet
from torch import optim

from .nnet import NNet


class UNet(NNet):
    def __init__(self):
        model = MonaiUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        optimizer = optim.Adam
        alpha = 0.015
        super().__init__("unet", model, optimizer, alpha)
