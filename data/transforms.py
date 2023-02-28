import monai.transforms as mt


def transform_function():
    """
    Define transform function

    Returns
    -------
    Transform
    """
    return mt.Compose(
        [
            mt.LoadImageD(keys=("flair", "seg")),  # Load NIFTI data
            mt.EnsureChannelFirstD(keys=("flair", "seg")),  # Make image and label channel-first
            mt.ScaleIntensityD(keys="flair"),  # Scale image intensity
            mt.ResizeD(
                ("flair", "seg"), (64, 64, 32), mode=("trilinear", "nearest")
            ),  # Resize images
        ]
    )
