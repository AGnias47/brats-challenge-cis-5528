#!/usr/bin/env python3

"""
run_model.py - Runs UNet and SegResNet models and displays output in a single script. Makes many assumptions and is not
paramaterized; mainly used for the purpose of generating a result for a report

References
----------
* https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
"""

from pathlib import Path

import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.networks.nets import UNet as MonaiUNet
import torch

import sys
sys.path.append(".")
from config import IMAGE_RESOLUTION
from data.containers import dataset_dict
from data.transforms import multi_channel_multiclass_label, validation_postprocessor

SLICES = 64
SLICES_TO_SHOW = 4
SLICE_GAP = SLICES // SLICES_TO_SHOW

if __name__ == "__main__":
    image_path = "local_data/train/BraTS2021_00000"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = MonaiUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    unet.load_state_dict(torch.load("trained_models/unet-model.pth"))
    unet.to(device)
    segresnet = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
    )
    segresnet.load_state_dict(torch.load("trained_models/segresnet-model.pth"))
    segresnet.to(device)
    data = dataset_dict(Path(image_path))
    transformed_image = multi_channel_multiclass_label()(data)
    image, label = transformed_image["image"].unsqueeze(0).to(device), transformed_image["seg"].to(device)
    with torch.no_grad():
        unet.eval()
        output = sliding_window_inference(
            inputs=image,
            roi_size=IMAGE_RESOLUTION,
            sw_batch_size=1,
            predictor=unet,
            overlap=0.5,
        )
        unet_output = validation_postprocessor()(output[0]).to("cpu")
        segresnet.eval()
        output = sliding_window_inference(
            inputs=image,
            roi_size=IMAGE_RESOLUTION,
            sw_batch_size=1,
            predictor=segresnet,
            overlap=0.5,
        )
        segresnet_output = validation_postprocessor()(output[0]).to("cpu")
    fig, ax = plt.subplots(SLICES_TO_SHOW - 1, 4, figsize=(8, 8))
    for i, s in enumerate(range(SLICE_GAP, SLICES - 1, SLICE_GAP)):
        ax[i, 0].imshow(transformed_image["image"][0, :, :, s], cmap="gray")
        if s == SLICE_GAP:
            ax[i, 0].set_title("Input Images\n")
        ax[i, 0].set_xlabel(f"Slice {s}")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])

        ax[i, 1].imshow(transformed_image["seg"][0, :, :, s].detach().cpu())
        if s == SLICE_GAP:
            ax[i, 1].set_title("Input Labels\n")
        ax[i, 1].set_xlabel(f"Slice {s}")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])

        ax[i, 2].imshow(unet_output[0, :, :, s].detach().cpu())
        if s == SLICE_GAP:
            ax[i, 2].set_title("UNet\n")
        ax[i, 2].set_xlabel(f"Slice {s}")
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])

        ax[i, 3].imshow(segresnet_output[0, :, :, s].detach().cpu())
        if s == SLICE_GAP:
            ax[i, 3].set_title("SegResNet\n")
        ax[i, 3].set_xlabel(f"Slice {s}")
        ax[i, 3].set_xticks([])
        ax[i, 3].set_yticks([])
    plt.show()
