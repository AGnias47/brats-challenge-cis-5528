#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import ignite
import monai
from monai.engines import SupervisedTrainer
from monai.handlers import MeanDice, from_engine, StatsHandler
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.transforms import AsDiscreteD
import torch
from torch.optim import Adadelta

from config import CPU, EPOCHS, GPU
from data.process_data import brats_dataloader

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
trainer = SupervisedTrainer(
    device=device,
    max_epochs=EPOCHS,
    train_data_loader=brats_dataloader("train"),
    network=unet,
    optimizer=Adadelta(unet.parameters(), 1e-5),
    loss_function=DiceLoss(to_onehot_y=True, softmax=True),
    prepare_batch=prepare_batch_fn,
    postprocessing=AsDiscreteD(
        keys=["flair", "seg"], argmax=(True, False), to_onehot=2
    ),
    key_train_metric={
        "train_meandice": MeanDice(output_transform=from_engine(["flair", "seg"]))
    },
    train_handlers=StatsHandler(
        tag_name="train_loss",
        output_transform=from_engine(["loss"], first=True),
        name="stats",
    ),
)
trainer.run()
