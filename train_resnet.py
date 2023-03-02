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
from nn.nn import train

if USE_SUMMARY_WRITER:
    from torch.utils.tensorboard import SummaryWriter


monai.utils.set_determinism(seed=42, additional_settings=None)

device = torch.device(GPU if torch.cuda.is_available() else CPU)
print(f"Device: {device}")
