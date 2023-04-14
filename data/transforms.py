import monai.transforms as mt
from monai.transforms import MapTransform
from config import IMAGE_RESOLUTION
import torch

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

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
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
            mt.LoadImageD(keys=("t1", "t1ce", "t2", "flair", "seg")),  # Load NIFTI data
            mt.EnsureChannelFirstD(keys=("t1", "t1ce", "t2", "flair")),  # Make image and label channel-first
            mt.EnsureTypeD(keys=("t1", "t1ce", "t2", "flair", "seg")),
            ConvertToMultiChannelBasedOnBratsClassesd("seg"),
            mt.OrientationD(keys=("t1", "t1ce", "t2", "flair", "seg"), axcodes="RAS"),
            mt.ScaleIntensityD(keys=("t1", "t1ce", "t2", "flair", "seg")),  # Scale image intensity
            mt.ConcatItemsD(keys=("t1", "t1ce", "t2", "flair"), name="image"),
            mt.ResizeD(
                ("image", "seg"),
                IMAGE_RESOLUTION,
                mode=("trilinear", "nearest-exact"),
            )
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
