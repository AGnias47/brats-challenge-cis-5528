from pathlib import Path
from torch.utils.data import random_split
from monai.data import PersistentDataset, Dataset, DataLoader, list_data_collate
from .data_transforms import transform_function
from config import BATCH_SIZE, LOCAL_DATA, WORKERS


def dataset_dicts(data_type="train"):
    """
    Generate a list of dicts indicating the location of data locally

    Parameters
    ----------
    data_type: str
        train or validation

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
    for subfolder in Path(LOCAL_DATA[data_type.casefold()]).iterdir():
        scan_name = subfolder.name
        data = {"name": scan_name}
        for scan_type in ["flair", "seg", "t1ce", "t1", "t2"]:
            data[scan_type] = f"{str(subfolder)}/{scan_name}_{scan_type}.nii.gz"
        dataset.append(data)
    return dataset


def brats_dataset(dataloader_type="train", persist=False):
    dataset = dataset_dicts(dataloader_type)
    data_transform_function = transform_function()
    if persist:
        return PersistentDataset(
            data=dataset,
            transform=data_transform_function,
            cache_dir=f"{LOCAL_DATA['cache']}/{dataloader_type.casefold()}",
        )

    else:
        return Dataset(data=dataset, transform=data_transform_function)


def brats_dataloader(dataloader_type="train"):
    """
    Creates a DataLoader object of the BraTS. Combines the use of dataset generation and transform function composition
    in the process of creating the DataLoader.

    Parameters
    ----------
    dataloader_type: str
        train or validation

    config.py values used
    ---------------------
    - LOCAL_DATA['cache']: str
        Directory where persistent data is stored; appends dataloader_type to the
        end of the given path
    - BATCH_SIZE: int
    - WORKERS: int

    Returns
    -------
    DataLoader
    """
    persistent_dataset = brats_dataset(dataloader_type)
    return DataLoader(
        dataset=persistent_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=True,
        collate_fn=list_data_collate,
    )


def train_test_val_dataloaders(train_ratio, test_ratio, val_ratio, dataloader_kwargs):
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
