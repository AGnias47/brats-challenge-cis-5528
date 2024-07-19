from pathlib import Path

from torch.utils.data import random_split
from monai.data import PersistentDataset, Dataset, DataLoader

from config import (
    LOCAL_DATA,
    PERSIST_DATASET,
    SCAN_TYPES,
    LABEL_KEY,
    SINGLE_CHANNEL_SCAN_TYPE,
    IMAGE_KEY,
)


def dataset_dicts(data_type="train", single_channel=False, dataset_path=None):
    """
    Generate a list of dicts indicating the location of data locally

    Parameters
    ----------
    data_type: str
        train: BraTS training data containing seg files
        validation: BraTS validation data without seg files
        other: will throw an error
    single_channel: bool
        If true, use single channel image; otherwise, use multi-channel image
    dataset_path: str
        Path to data; if not defined, uses LOCAL_DATA[data_type]


    config.py values used
    ---------------------
    - LOCAL_DATA[data_type]: str
        Location of dataset directories

    Returns
    -------
    list of dict of the form
        name: name of subdirectory containing NIFTI files, ex. t1c: path to t1c scan
    """
    dataset = list()
    if single_channel:
        scan_types = [SINGLE_CHANNEL_SCAN_TYPE]
    else:
        scan_types = SCAN_TYPES
    if data_type.casefold() == "train":
        scan_types = SCAN_TYPES + [LABEL_KEY]
    if not dataset_path:
        dataset_path = LOCAL_DATA[data_type.casefold()]
    for subfolder in Path(dataset_path).iterdir():
        dataset.append(dataset_dict(subfolder, scan_types))
    if single_channel:
        for d in dataset:
            d[IMAGE_KEY] = d.pop(SINGLE_CHANNEL_SCAN_TYPE)
    return dataset


def dataset_dict(subfolder, scan_types=None):
    if not scan_types:
        scan_types = SCAN_TYPES + [LABEL_KEY]
    scan_name = subfolder.name
    data = {"name": scan_name}
    for scan_type in scan_types:
        data[scan_type] = f"{str(subfolder)}/{scan_name}-{scan_type}.nii.gz"
    return data


def brats_dataset(
    data_type, transform_function, single_channel=False, dataset_path=None
):
    """
    Returns a BraTS Dataset object

    Parameters
    ----------
    data_type: str
        train: BraTS training data containing seg files
        validation: BraTS validation data without seg files
        other: will throw an error
    transform_function: callable
    single_channel: bool
        If true, use single channel image; otherwise, use multi-channel image
    dataset_path: str
        Path to data; if not defined, uses LOCAL_DATA[data_type]

    config.py values used
    ---------------------
    - PERSIST_DATASET: bool
        If true, use a PersistentDataset object. Only use when Dataset data is stable, or going to be unchanged for a
        significant period of time
    - LOCAL_DATA['cache']: str
        If dataset is set to persist, directory where persistent data is stored; appends dataloader_type to the
        end of the given path

    References
    ----------
    * https://stackoverflow.com/questions/54637847/how-to-change-dictionary-keys-in-a-list-of-dictionaries

    Returns
    -------
    Dataset
    """
    dataset = dataset_dicts(data_type, single_channel, dataset_path)
    data_transform_function = transform_function()
    if PERSIST_DATASET:
        return PersistentDataset(
            data=dataset,
            transform=data_transform_function,
            cache_dir=f"{LOCAL_DATA['cache']}/{data_type.casefold()}",
        )
    return Dataset(data=dataset, transform=data_transform_function)


def train_test_val_dataloaders(
    train_ratio,
    test_ratio,
    val_ratio,
    dataloader_kwargs,
    transform_function,
    single_channel=False,
    dataset_path=None,
):
    """
    Creates a train, test, and validation DataLoader objects of the BraTS dataset. Combines the use of dataset
    generation and transform function composition in the process of creating the DataLoaders.

    Parameters
    ----------
    train_ratio: float
    test_ratio: float
    val_ratio: float
    dataloader_kwargs: dict
    transform_function: callable
    single_channel: bool
        If true, use single channel image; otherwise, use multi-channel image
    dataset_path: str
        Path to data; if not defined, uses LOCAL_DATA[data_type]

    Returns
    -------
    DataLoader
    """
    ratio_total = train_ratio + test_ratio + val_ratio
    if ratio_total < 0.99 or ratio_total > 1.01:
        raise ValueError("Invalid train-test-val ratios provided")
    dataset = brats_dataset("train", transform_function, single_channel, dataset_path)
    train, test, val = random_split(dataset, [train_ratio, test_ratio, val_ratio])
    return (
        DataLoader(train, **dataloader_kwargs),
        DataLoader(test, **dataloader_kwargs),
        DataLoader(val, **dataloader_kwargs),
    )
