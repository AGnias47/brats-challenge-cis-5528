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
from data import transforms
from nn.unet import UNet
from nn.segresnet import SegResNet

DEFAULT_EPOCHS = 150


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(name)-15s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
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

    if SINGLE_CHANNEL:
        transform_function = transforms.single_channel_binary_label
        input_channels = 1
        output_channels = 1
    else:
        transform_function = transforms.multi_channel_multiclass_label
        input_channels = 4
        output_channels = 3

    if args.model.casefold() == "unet":
        nnet = UNet(
            input_channels=input_channels,
            output_channels=output_channels,
            learning_rate=LEARNING_RATE["unet"],
        )
    elif args.model.casefold() == "segresnet":
        nnet = SegResNet(
            input_channels=input_channels,
            output_channels=output_channels,
            learning_rate=LEARNING_RATE["segresnet"],
        )
    else:
        raise ValueError("Invalid model type specified")

    train_dataloader, test_dataloader, validation_dataloader = train_test_val_dataloaders(
        train_ratio=TRAIN_RATIO,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        dataloader_kwargs=dataloader_kwargs,
        transform_function=transform_function,
        single_channel=SINGLE_CHANNEL,
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
