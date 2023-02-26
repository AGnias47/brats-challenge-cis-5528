#!/usr/bin/env python3

"""
Resources
---------
* https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T#scrollTo=VdFXzJV-oNEM
* https://github.com/LucasFidon/TRABIT_BraTS2021/blob/main/src/data/brats21_dataset.py
"""

import matplotlib.pyplot as plt
from data.process_data import brats_dataloader


data_loader = brats_dataloader()
data = next(iter(data_loader))
print("Image size in DataLoader, Label size in DataLoader")
print(data["flair"].shape, data["seg"].shape)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(data["flair"][0, 0, :, :, 16])
ax[1].imshow(data["seg"][0, 0, :, :, 16])
plt.show()
