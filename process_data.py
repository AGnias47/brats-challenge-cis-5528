#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import monai
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import torch
from torch.optim import Adadelta
from torch.optim.lr_scheduler import ExponentialLR

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.process_data import train_test_val_dataloaders
from nn.nn import train

monai.utils.set_determinism(seed=42, additional_settings=None)

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
dataloader_kwargs = DATALOADER_KWARGS_GPU if device == "GPU" else DATALOADER_KWARGS_CPU
train_dataloader, test_dataloader, validation_dataloader = train_test_val_dataloaders(
    TRAIN_RATIO, TEST_RATIO, VAL_RATIO, dataloader_kwargs
)
model = train(
    device,
    unet,
    loss_function,
    optimizer,
    scheduler,
    train_dataloader,
)
