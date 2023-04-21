import monai.transforms as mt
from monai.transforms import MapTransform
from config import IMAGE_RESOLUTION
import torch

from config import IMAGE_KEY, LABEL_KEY, SCAN_TYPES


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


class OneHotLabeling(MapTransform):
    """
    Converts the segmentation into a usable one-hot format.

    From: https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

    Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edematous/invaded tissue (ED — label 2),
    and the necrotic tumor core (NCR — label 1), as described in the latest BraTS summarizing paper. The ground truth
    data were created after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to
    the same resolution (1 mm3) and skull-stripped.

    Resources
    ---------
    https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = [d[key] == 4, d[key] == 2, d[key] == 1]
            d[key] = torch.stack(result, axis=0).float()
        return d


def single_channel_binary_label():
    """
    Transform function for single class image and binary label

    Returns
    -------
    Transform
    """
    return mt.Compose(
        [
            mt.LoadImageD(keys=(IMAGE_KEY, LABEL_KEY)),  # Load NIFTI data
            MultiToBinary(keys=LABEL_KEY),
            mt.EnsureChannelFirstD(
                keys=(IMAGE_KEY, LABEL_KEY)
            ),  # Make image and label channel-first
            mt.ScaleIntensityD(keys=IMAGE_KEY),  # Scale image intensity
            mt.ResizeD(
                (IMAGE_KEY, LABEL_KEY),
                IMAGE_RESOLUTION,
                mode=("trilinear", "nearest-exact"),
            ),  # Resize images
        ]
    )


def multi_channel_binary_label():
    """
    Transform function for multi-class image and binary label

    Returns
    -------
    Transform
    """
    return mt.Compose(
        [
            mt.LoadImageD(keys=(*SCAN_TYPES, LABEL_KEY)),  # Load NIFTI data
            MultiToBinary(LABEL_KEY),
            mt.EnsureChannelFirstD(
                keys=(*SCAN_TYPES, LABEL_KEY)
            ),  # Make image and label channel-first
            mt.EnsureTypeD(keys=(*SCAN_TYPES, LABEL_KEY)),
            mt.OrientationD(keys=SCAN_TYPES, axcodes="RAS"),
            mt.ScaleIntensityD(keys=SCAN_TYPES),  # Scale image intensity
            mt.ConcatItemsD(keys=SCAN_TYPES, name=IMAGE_KEY),
            mt.ResizeD(
                (IMAGE_KEY, LABEL_KEY),
                IMAGE_RESOLUTION,
                mode=("trilinear", "nearest-exact"),
            ),
        ]
    )


def multi_channel_multiclass_label():
    """
    Transform function for multi-class image and multi-class label

    Returns
    -------
    Transform
    """
    return mt.Compose(
        [
            mt.LoadImageD(keys=(*SCAN_TYPES, LABEL_KEY)),  # Load NIFTI data
            mt.EnsureChannelFirstD(
                keys=SCAN_TYPES
            ),  # Make image and label channel-first
            mt.EnsureTypeD(keys=(*SCAN_TYPES, LABEL_KEY)),
            OneHotLabeling(LABEL_KEY),
            mt.OrientationD(keys=SCAN_TYPES, axcodes="RAS"),
            mt.ScaleIntensityD(keys=SCAN_TYPES),  # Scale image intensity
            mt.ConcatItemsD(keys=SCAN_TYPES, name=IMAGE_KEY),
            mt.ResizeD(
                (IMAGE_KEY, LABEL_KEY),
                IMAGE_RESOLUTION,
                mode=("trilinear", "nearest-exact"),
            ),
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
