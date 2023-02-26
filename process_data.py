#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import matplotlib.pyplot as plt
from monai.data import PersistentDataset, DataLoader

from config import BATCH_SIZE, WORKERS
from data.process_data import dataset_dicts
from data.data_transforms import transform_function
from helpers.utils import seed_everything


seed_everything(42)
dataset = dataset_dicts()
data_transform_function = transform_function()
data_dict_transform = data_transform_function(dataset[0])
print(data_dict_transform["flair"].shape, data_dict_transform["seg"].shape)
persistent_dataset = PersistentDataset(
    dataset, data_transform_function, "local_data/persistent_dataset/train"
)
data_loader = DataLoader(
    persistent_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS
)
data = next(iter(data_loader))
print("Image size in DataLoader, Label size in DataLoader")
print(data["flair"].shape, data["seg"].shape)
# View DataLoader contents
fig, ax = plt.subplots(1, 2)
ax[0].imshow(data["flair"][0, 0, :, :, 16])
ax[1].imshow(data["seg"][0, 0, :, :, 16])
plt.show()
