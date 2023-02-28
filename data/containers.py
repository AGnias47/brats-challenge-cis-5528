from pathlib import Path
from torch.utils.data import random_split
from monai.data import PersistentDataset, Dataset, DataLoader
from .transforms import transform_function
from config import LOCAL_DATA, PERSIST_DATASET


def dataset_dicts(data_type="train"):
    """
    Generate a list of dicts indicating the location of data locally

    Parameters
    ----------
    data_type: str
        train: BraTS training data containing seg files
        validation: BraTS validation data without seg files
        other: will throw an error

    config.py values used
    ---------------------
    - LOCAL_DATA[data_type]: str
        Location of dataset directories

    Returns
    -------
    list of dict of the form
        name: name of subdirectory containing NIFTI files
        flair: path to flair scan
        seg: path to seg scan
        t1ce: path to t1ce scan
        t1: path to t1 scan
        t2: path to t2 scan
    """
    dataset = list()
    scan_types = ["flair", "t1ce", "t1", "t2"]
    if data_type.casefold() == "train":
        scan_types.append("seg")
    for subfolder in Path(LOCAL_DATA[data_type.casefold()]).iterdir():
        scan_name = subfolder.name
        data = {"name": scan_name}
        for scan_type in scan_types:
            data[scan_type] = f"{str(subfolder)}/{scan_name}_{scan_type}.nii.gz"
        dataset.append(data)
    return dataset


def brats_dataset(data_type):
    """
    Returns a BraTS Dataset object

    Parameters
    ----------
    data_type: str
        train: BraTS training data containing seg files
        validation: BraTS validation data without seg files
        other: will throw an error

    config.py values used
    ---------------------
    - PERSIST_DATASET: bool
        If true, use a PersistentDataset object. Only use when Dataset data is stable, or going to be unchanged for a
        significant period of time
    - LOCAL_DATA['cache']: str
        If dataset is set to persist, directory where persistent data is stored; appends dataloader_type to the
        end of the given path

    Returns
    -------
    Dataset
    """
    dataset = dataset_dicts(data_type)
    data_transform_function = transform_function()
    if PERSIST_DATASET:
        return PersistentDataset(
            data=dataset,
            transform=data_transform_function,
            cache_dir=f"{LOCAL_DATA['cache']}/{data_type.casefold()}",
        )
    else:
        return Dataset(data=dataset, transform=data_transform_function)


def train_test_val_dataloaders(train_ratio, test_ratio, val_ratio, dataloader_kwargs):
    """
    Creates a train, test, and validation DataLoader objects of the BraTS dataset. Combines the use of dataset
    generation and transform function composition in the process of creating the DataLoaders.

    Parameters
    ----------
    train_ratio: float
    test_ratio: float
    val_ratio: float
    dataloader_kwargs: dict

    Returns
    -------
    DataLoader
    """
    ratio_total = train_ratio + test_ratio + val_ratio
    if ratio_total < 0.99 or ratio_total > 1.01:
        raise ValueError("Invalid train-test-val ratios provided")
    dataset = brats_dataset("train")
    train, test, val = random_split(dataset, [train_ratio, test_ratio, val_ratio])
    return (
        DataLoader(train, **dataloader_kwargs),
        DataLoader(test, **dataloader_kwargs),
        DataLoader(val, **dataloader_kwargs),
    )