#!/usr/bin/env python3

"""
tune_hyperparameters.py - Uses Optuna to determine ideal model hyperparameters
"""

import argparse

import monai
from monai.networks.nets import UNet, SegResNet
import optuna
from optuna.integration.mlflow import MLflowCallback
import torch

from config import *  # pylint: disable=wildcard-import,unused-wildcard-import
from data.containers import train_test_val_dataloaders
from nn.optunet import Optunet


class OptunaUnet(Optunet):
    def __init__(self, trial):
        name = "optuna_unet"
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        super().__init__(name, trial, model)


class OptunaSegResNet(Optunet):
    def __init__(self, trial):
        name = "optuna_segresnet"
        model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
        )
        super().__init__(name, trial, model)


def unet_objective(trial):
    model = OptunaUnet(trial)
    image_key = "flair"  # trial.suggest_categorical("image_key", ["flair", "t1ce", "t1", "t2"])
    train, _, val = train_test_val_dataloaders(TRAIN_RATIO, TEST_RATIO, VAL_RATIO, dataloader_kwargs, image_key, "seg")
    return model.run_training(
        train,
        val,
        args.epochs,
    )


def segresnet_objective(trial):
    model = OptunaSegResNet(trial)
    image_key = "flair"  # trial.suggest_categorical("image_key", ["flair", "t1ce", "t1", "t2"])
    train, _, val = train_test_val_dataloaders(TRAIN_RATIO, TEST_RATIO, VAL_RATIO, dataloader_kwargs, image_key, "seg")
    return model.run_training(
        train,
        val,
        args.epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-t", "--trials", type=int, default=20)
    args = parser.parse_args()
    monai.utils.set_determinism(seed=42, additional_settings=None)
    if not torch.cuda.is_available():
        print("WARNING: GPU is not available")
        dataloader_kwargs = DATALOADER_KWARGS_CPU
    else:
        print("Using GPU")
        dataloader_kwargs = DATALOADER_KWARGS_GPU

    if "unet" in args.model.casefold():
        objective = unet_objective
        study_name = "UNet Hyperparameter Optimization"
    elif "segresnet" in args.model.casefold():
        objective = segresnet_objective
        study_name = "SegResNet Hyperparameter Optimization"
    else:
        raise ValueError("Invalid model type specified")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    try:
        study.optimize(
            objective,
            callbacks=[MLflowCallback(metric_name="validation_mean_dice")],
            n_trials=args.trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        pass
    print("Optuna study best trial:")
    trial = study.best_trial
    auc = trial.value
    print(f"Value: {auc}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
