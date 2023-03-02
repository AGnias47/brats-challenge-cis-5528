from monai.losses import DiceCELoss
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
        loss_function = DiceCELoss(sigmoid=True)
        optimizer = optim.Adam
        alpha = 1e-3
        gamma = 1e-3
        super().__init__(model, loss_function, optimizer, alpha, gamma)
