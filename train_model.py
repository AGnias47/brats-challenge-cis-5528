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
from monai.losses import DiceLoss
from monai.transforms import Compose, Activations, AsDiscrete
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders
from nn.nn import train


monai.utils.set_determinism(seed=42, additional_settings=None)

device = torch.device(GPU if torch.cuda.is_available() else CPU)
unet_model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(unet_model.parameters(), 1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.001)
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
with SummaryWriter() as summary_writer:
    best_model_wts = train(
        device=device,
        model=unet_model,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        validation_metric=validation_metric,
        validation_postprocessor=validation_postprocessor,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        summary_writer=summary_writer,
        epochs=EPOCHS,
    )

torch.save(best_model_wts, "trained-models/unet-model")
