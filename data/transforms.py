import monai.transforms as mt
from config import IMAGE_RESOLUTION, RESIZING_ALGORITHM


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
            mt.EnsureChannelFirstD(
                keys=("flair", "seg")
            ),  # Make image and label channel-first
            mt.ScaleIntensityD(keys="flair"),  # Scale image intensity
            mt.ResizeD(
                ("flair", "seg"), IMAGE_RESOLUTION, mode=RESIZING_ALGORITHM
            ),  # Resize images
        ]
    )
