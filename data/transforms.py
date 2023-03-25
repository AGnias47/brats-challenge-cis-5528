import monai.transforms as mt
from monai.transforms import MapTransform
from config import IMAGE_RESOLUTION


class MultiToBinary(MapTransform):
    """
    Converts a multiclass segmentation into a binary class segmentation by setting any class value other than 0 to 1

    Resources
    ---------
    * https://stackoverflow.com/a/60095396/8728749
    * https://stackoverflow.com/questions/58002836/pytorch-1-if-x-0-5-else-0-for-x-in-outputs-with-tensors
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = (d[key] > 1).float()
            d[key] = result
        return d


def dict_transform_function():
    """
    Transform function for data stored in a dict

    Returns
    -------
    Transform
    """
    return mt.Compose(
        [
            mt.LoadImageD(keys=("image", "label")),  # Load NIFTI data
            MultiToBinary(keys="label"),
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
    """
    Transform function for a single image

    Returns
    -------
    Transform
    """
    return mt.Compose(
        [
            mt.LoadImage(image_only=True, ensure_channel_first=True),  # Load NIFTI data
            mt.ScaleIntensity(),  # Scale image intensity
            mt.Resize(IMAGE_RESOLUTION),  # Resize images
        ]
    )


def validation_postprocessor():
    """
    Transform function for post-processing segmentation

    Returns
    -------
    Transform
    """
    return mt.Compose([mt.Activations(sigmoid=True), mt.AsDiscrete(threshold=0.5)])
