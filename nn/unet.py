from monai.networks.nets import UNet as MonaiUNet
from torch import optim

from .nnet import NNet


class UNet(NNet):
    def __init__(self, input_channels=4, output_channels=3, learning_rate=0.15):
        model = MonaiUNet(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=output_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        optimizer = optim.Adam
        super().__init__("unet", model, optimizer, learning_rate)
