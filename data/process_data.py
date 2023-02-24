from pathlib import Path


def dataset_filepaths(directory="local_data/train"):
    """
    Generate a list of dicts indicating the location of data locally

    Parameters
    ----------
    directory: str

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
    for subfolder in Path(directory).iterdir():
        scan_name = subfolder.name
        data = {"name": scan_name}
        for scan_type in ["flair", "seg", "t1ce", "t1", "t2"]:
            data[scan_type] = f"{str(subfolder)}/{scan_name}_{scan_type}.nii.gz"
        dataset.append(data)
    return dataset
