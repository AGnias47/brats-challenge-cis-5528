#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""
from copy import deepcopy

import monai
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import Compose, Activations, AsDiscrete
from monai.visualize import plot_2d_or_3d_image
import torch
from torch.optim import Adadelta
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.process_data import train_test_val_dataloaders


monai.utils.set_determinism(seed=42, additional_settings=None)

device = torch.device(GPU if torch.cuda.is_available() else CPU)
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = DiceLoss(sigmoid=True)
optimizer = Adadelta(model.parameters(), 1e-5)
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
summary_writer = SummaryWriter()
best_model_wts = deepcopy(model.state_dict())
best_metric = -torch.inf
for epoch in tqdm(range(EPOCHS), desc=f"Training over {EPOCHS} epochs"):
    running_loss = float(0)
    model.train()
    for batch in train_dataloader:
        image, label = batch["flair"].to(device), batch["seg"].to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(image)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * image.size(0)
    total_loss = running_loss / (len(train_dataloader.dataset))
    scheduler.step()
    print(f"Epoch {epoch} Training Loss: {total_loss:.4f}")
    print("-" * 25)
    model.eval()
    with torch.no_grad():
        for batch in validation_dataloader:
            image, label = batch["flair"].to(device), batch["seg"].to(device)
            roi_size = (96, 96, 96)
            output = sliding_window_inference(image, roi_size, 4, model)
            output = [validation_postprocessor(i) for i in decollate_batch(output)]
            metric = validation_metric(y_pred=output, y=label)
        metric_result = metric.aggregate.item()
        metric.reset()
        print(f"Epoch {epoch} Mean Dice: {metric_result:.4f}")
        print("-" * 25)
        best_model_wts = None
        if metric > best_metric:
            best_metric = metric
            best_model_wts = deepcopy(model.state_dict())
            plot_2d_or_3d_image(image, epoch + 1, summary_writer, index=0, tag="image")
            plot_2d_or_3d_image(label, epoch + 1, summary_writer, index=0, tag="label")
            plot_2d_or_3d_image(output, epoch + 1, summary_writer, index=0, tag="output")

torch.save(best_model_wts, "trained-models/unet-model")
