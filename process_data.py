#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import ignite
import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import torch
from torch.optim import Adadelta
from torch.optim.lr_scheduler import ExponentialLR

from config import CPU, EPOCHS, GPU
from data.process_data import brats_dataloader
from nn.nn import train

monai.utils.set_determinism(seed=42, additional_settings=None)


def prepare_batch_fn(batch, device, non_blocking):
    return (
        ignite.utils.convert_tensor(batch["flair"], device, non_blocking),
        ignite.utils.convert_tensor(batch["seg"], device, non_blocking),
    )


device = torch.device(GPU if torch.cuda.is_available() else CPU)
unet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceLoss(sigmoid=True)
optimizer = Adadelta(unet.parameters(), 1e-5)
scheduler = ExponentialLR(optimizer, gamma=0.001)
data = brats_dataloader("train")
model = train(
    device,
    unet,
    loss_function,
    optimizer,
    scheduler,
    data,
)
