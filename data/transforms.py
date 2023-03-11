import monai.transforms as mt
from config import IMAGE_RESOLUTION


def dict_transform_function():
    """
    Define transform function

    Returns
    -------
    Transform
    """
    return mt.Compose(
        [
            mt.LoadImageD(keys=("image", "label")),  # Load NIFTI data
            mt.EnsureChannelFirstD(keys=("image", "label")),  # Make image and label channel-first
            mt.ScaleIntensityD(keys="image"),  # Scale image intensity
            mt.ResizeD(
                ("image", "label"),
                IMAGE_RESOLUTION,
                mode=("trilinear", "nearest-exact"),
            ),  # Resize images
        ]
    )


def single_image_transform_function():
    return mt.Compose(
        [
            mt.LoadImage(image_only=True, ensure_channel_first=True),  # Load NIFTI data
            mt.ScaleIntensity(),  # Scale image intensity
            mt.Resize(IMAGE_RESOLUTION),  # Resize images
        ]
    )


def validation_postprocessor():
    return mt.Compose([mt.Activations(sigmoid=True), mt.AsDiscrete(threshold=0.5)])
