from monai.networks.nets import UNet as MonaiUNet
from torch import optim

from .nnet import NNet


class UNet(NNet):
    def __init__(self):
        self.name = "unet"
        model = MonaiUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        optimizer = optim.ASGD
        alpha = 0.0008040631180475624
        gamma = 7.328377594951827e-05
        super().__init__(model, optimizer, alpha, gamma)
