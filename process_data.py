#!/usr/bin/env python3

"""
Resources
---------
https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

from monai.data import PersistentDataset

from data.process_data import dataset_dicts
from data.data_transforms import transform_function


dataset = dataset_dicts()
data_transform_function = transform_function()
data_dict_transform = data_transform_function(dataset[0])
print(data_dict_transform["flair"].shape, data_dict_transform["seg"].shape)
persistent_dataset = PersistentDataset(dataset, data_transform_function, "local_data/persistent_dataset/train")
