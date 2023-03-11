#!/usr/bin/env python3

"""

Refs
https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
"""
import argparse

import matplotlib.pyplot as plt
import monai
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet as MonaiUNet
import torch

from data.transforms import single_image_transform_function, validation_postprocessor
from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, default="trained_models/unet-model.pth"
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default="local_data/validation/BraTS2021_00001/BraTS2021_00001_t1ce.nii.gz",
    )
    args = parser.parse_args()
    monai.utils.set_determinism(seed=42, additional_settings=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MonaiUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    transformed_image = single_image_transform_function()(args.image_path)
    image = transformed_image.unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        output = sliding_window_inference(image, (96, 96, 96), 1, model)
        processed_output = validation_postprocessor()(output[0]).to("cpu")
    slices = 64
    slices_to_show = 4
    slice_gap = slices // slices_to_show
    fig, ax = plt.subplots(slices_to_show-1, 2, figsize=(5, 8))
    for i, s in enumerate(range(slice_gap, slices-1, slice_gap)):
        ax[i, 0].imshow(transformed_image[0, :, :, s], cmap="gray")
        if s == slice_gap:
            ax[i, 0].set_title(f"Input Images\n")
        ax[i, 0].set_xlabel(f"Slice {s}")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].imshow(processed_output[0, :, :, s].detach().cpu())
        if s == slice_gap:
            ax[i, 1].set_title(f"Segmentations\n")
        ax[i, 1].set_xlabel(f"Slice {s}")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
    plt.show()
