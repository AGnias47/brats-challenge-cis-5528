#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import argparse

import monai
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Compose, Activations, AsDiscrete
import torch
from torch import optim

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders
from nn.nnet import NNet

if USE_SUMMARY_WRITER:
    from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", help="Neural Network type to use; one of [unet]", default="unet"
)
parser.add_argument(
    "-e", "--epochs", help="Number of training epochs to use", type=int, default=5
)
args = parser.parse_args()

monai.utils.set_determinism(seed=42, additional_settings=None)
if not torch.cuda.is_available():
    print("WARNING: GPU is not available; continue?")
    dataloader_kwargs = DATALOADER_KWARGS_CPU
else:
    print("Using GPU")
    dataloader_kwargs = DATALOADER_KWARGS_GPU

if args.model.casefold() == "unet":
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
else:
    raise ValueError("Invalid model type specified")

nnet = NNet(model, DiceCELoss(sigmoid=True), optim.Adam, alpha=1e-3, gamma=1e-3)

train_dataloader, test_dataloader, validation_dataloader = train_test_val_dataloaders(
    TRAIN_RATIO, TEST_RATIO, VAL_RATIO, dataloader_kwargs
)
validation_metric = DiceMetric(
    include_background=True, reduction="mean", get_not_nans=False
)
validation_postprocessor = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

if USE_SUMMARY_WRITER:
    with SummaryWriter(LOCAL_DATA["tensorboard_logs"]) as summary_writer:
        nnet.run_training(
            train_dataloader,
            validation_dataloader,
            validation_postprocessor,
            validation_metric,
            args.epochs,
            summary_writer,
        )
else:
    nnet.run_training(
        train_dataloader,
        validation_dataloader,
        validation_postprocessor,
        validation_metric,
        args.epochs,
    )

nnet.save_model(f"{LOCAL_DATA['model_output']}/unet-model.pth")
