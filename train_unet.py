#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import monai
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.transforms import Compose, Activations, AsDiscrete
import torch
from torch.optim.lr_scheduler import ExponentialLR

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders
from model_params import UNET
from nn.nnet import NNet

if USE_SUMMARY_WRITER:
    from torch.utils.tensorboard import SummaryWriter


monai.utils.set_determinism(seed=42, additional_settings=None)

device = torch.device(GPU if torch.cuda.is_available() else CPU)
print(f"Device: {device}")
unet_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = DiceCELoss(sigmoid=True)
optimizer = UNET["optimizer"](unet_model.parameters(), UNET["alpha"])
scheduler = ExponentialLR(optimizer, gamma=UNET["gamma"])
validation_metric = DiceMetric(
    include_background=True, reduction="mean", get_not_nans=False
)
validation_postprocessor = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
dataloader_kwargs = DATALOADER_KWARGS_GPU if device == "GPU" else DATALOADER_KWARGS_CPU
train_dataloader, test_dataloader, validation_dataloader = train_test_val_dataloaders(
    TRAIN_RATIO, TEST_RATIO, VAL_RATIO, dataloader_kwargs
)

nnet = NNet(unet_model, loss_function, optimizer, scheduler)


if USE_SUMMARY_WRITER:
    with SummaryWriter() as summary_writer:
        nnet.run_training(
            train_dataloader,
            validation_dataloader,
            validation_postprocessor,
            validation_metric,
            EPOCHS,
            summary_writer,
        )
else:
    nnet.run_training(
        train_dataloader,
        validation_dataloader,
        validation_postprocessor,
        validation_metric,
        EPOCHS,
    )

nnet.save_model(f"{LOCAL_DATA['model_output']}/unet-model.pth")
