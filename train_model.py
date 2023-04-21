#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import argparse
import logging

import monai
import torch
from torch.utils.tensorboard import SummaryWriter

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders
from data.transforms import multi_channel_multiclass_label
from nn.unet import UNet
from nn.segresnet import SegResNet

DEFAULT_EPOCHS = 150


logging.basicConfig(format="%(asctime)s %(name)-15s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="Neural Network type to use; one of [unet, segresnet] (default is unet)",
    default="unet",
)
parser.add_argument(
    "-e",
    "--epochs",
    help=f"Number of training epochs to use (default is {DEFAULT_EPOCHS})",
    type=int,
    default=DEFAULT_EPOCHS,
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
elif args.model.casefold() == "segresnet":
    nnet = SegResNet()
else:
    raise ValueError("Invalid model type specified")

train_dataloader, test_dataloader, validation_dataloader = train_test_val_dataloaders(
    TRAIN_RATIO, TEST_RATIO, VAL_RATIO, dataloader_kwargs, multi_channel_multiclass_label
)

try:
    with SummaryWriter(LOCAL_DATA["tensorboard_logs"]) as summary_writer:
        nnet.run_training(
            train_dataloader,
            validation_dataloader,
            args.epochs,
            summary_writer,
        )
except KeyboardInterrupt:
    pass  # Allow us to end early and still test

f1 = nnet.test(test_dataloader, summary_writer)
print(f"F1 score: {f1}")
