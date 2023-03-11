#!/usr/bin/env python3

import argparse

import monai
from monai.networks.nets import UNet as MonaiUNet
import torch

from data.transforms import single_image_transform_function
from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, default="model-output/unet-model.pth"
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
    image = single_image_transform_function()(args.image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        model.eval()
        output = model(image)

