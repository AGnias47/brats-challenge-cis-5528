#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np

"""
https://stackoverflow.com/questions/21444338/transpose-nested-list-in-python
"""


class Results:
    def __init__(self):
        self.loss = None
        self.validation = None


unet_results = Results()
segresnet_results = Results()

with open("results/data/multi-channel/unet-training-loss.json") as UL:
    unet_results.loss = np.array(json.load(UL)).T.tolist()

with open("results/data/multi-channel/unet-validation-mean-dice.json") as UV:
    unet_results.validation = np.array(json.load(UV)).T.tolist()

with open("results/data/multi-channel/segresnet-training-loss.json") as SL:
    segresnet_results.loss = np.array(json.load(SL)).T.tolist()

with open("results/data/multi-channel/segresnet-validation-mean-dice.json") as SV:
    segresnet_results.validation = np.array(json.load(SV)).T.tolist()

fig, ax = plt.subplots(2, figsize=(10, 10))
ax[0].set_title("Training Loss")
ax[0].plot(unet_results.loss[1], unet_results.loss[2], label="Residual U-Net", color="#ff7f0e")
ax[0].plot(segresnet_results.loss[1], segresnet_results.loss[2], label="SegResNet")
ax[1].set_xlabel("Epoch")
ax[1].set_title("Validation Score")
ax[1].plot(
    unet_results.validation[1],
    unet_results.validation[2],
    label="Residual U-Net",
    color="#ff7f0e",
)
ax[1].plot(segresnet_results.validation[1], segresnet_results.validation[2], label="SegResNet")
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right")

plt.show()
