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
        optimizer = optim.Adam
        alpha = 2.1433553525451694e-05
        gamma = 4.182657295694138e-05
        super().__init__(model, optimizer, alpha, gamma)
