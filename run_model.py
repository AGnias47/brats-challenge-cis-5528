#!/usr/bin/env python3

"""
run_model.py - Runs segmentation on a trained model

References
----------
* https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import monai
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.networks.nets import UNet as MonaiUNet
import torch

from config import IMAGE_RESOLUTION
from data.containers import dataset_dict
from data.transforms import dict_transform_function, validation_postprocessor

SLICES = 64
SLICES_TO_SHOW = 4
SLICE_GAP = SLICES // SLICES_TO_SHOW

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default="trained_models/unet-model.pth")
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default="local_data/train/BraTS2021_00000",
    )
    args = parser.parse_args()
    monai.utils.set_determinism(seed=42, additional_settings=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "unet" in args.model_path.casefold():
        model = MonaiUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif "resnet" in args.model_path.casefold():
        model = SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
        )
    else:
        raise ValueError("Invalid model type specified")
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    data = dataset_dict(Path(args.image_path))
    transformed_image = dict_transform_function()(data)
    image, label = transformed_image["image"].unsqueeze(0).to(device), transformed_image["seg"].to(device)
    with torch.no_grad():
        model.eval()
        output = sliding_window_inference(
            inputs=image,
            roi_size=IMAGE_RESOLUTION,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
        processed_output = validation_postprocessor()(output[0]).to("cpu")
    fig, ax = plt.subplots(SLICES_TO_SHOW - 1, 3, figsize=(8, 8))
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

        ax[i, 2].imshow(processed_output[0, :, :, s].detach().cpu())
        if s == SLICE_GAP:
            ax[i, 2].set_title("Model Segmentations\n")
        ax[i, 2].set_xlabel(f"Slice {s}")
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
    plt.show()
