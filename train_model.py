#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import argparse

import monai
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
import torch

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders
from nn.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", help="Neural Network type to use; one of [unet]", default="unet"
)
parser.add_argument(
    "-e", "--epochs", help="Number of training epochs to use", type=int, default=5
)
parser.add_argument(
    "-s",
    "--use-summary-writer",
    help="Flag to log data to Tensorboard (default is True)",
    type=bool,
    default=True,
)
args = parser.parse_args()

monai.utils.set_determinism(seed=42, additional_settings=None)
if not torch.cuda.is_available():
    print("WARNING: GPU is not available")
    dataloader_kwargs = DATALOADER_KWARGS_CPU
else:
    print("Using GPU")
    dataloader_kwargs = DATALOADER_KWARGS_GPU

if args.model.casefold() == "unet":
    nnet = UNet()
else:
    raise ValueError("Invalid model type specified")

train_dataloader, test_dataloader, validation_dataloader = train_test_val_dataloaders(
    TRAIN_RATIO, TEST_RATIO, VAL_RATIO, dataloader_kwargs
)
validation_metric = DiceMetric(
    include_background=True, reduction="mean", get_not_nans=False
)
validation_postprocessor = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

if args.use_summary_writer:
    from torch.utils.tensorboard import SummaryWriter

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

nnet.save_model(f"{LOCAL_DATA['model_output']}/{nnet.name}-model.pth")
