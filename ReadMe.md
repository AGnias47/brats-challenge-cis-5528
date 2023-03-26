# CIS 5528 BraTS Project

Repository for all code related to project.

## Local Setup

### Python Setup

* Install pyenv via instructions at https://github.com/pyenv/pyenv-virtualenv and update .*rc file accordingly
* Install Python 3.9 `pyenv install 3.9.16`
* Create venv `pyenv virtualenv 3.9.16 brats`
* Activate venv `pyenv local brats`
* Install dependencies `pip install -r requirements.txt`

### Download Data

* Download zip of training data from https://drive.google.com/file/d/1TQpeHXTB4YuwNJL0A5oCtVXZ3CnPa4SR/view?usp=share_link
* Move the file to `./local_data`
* Unzip `unzip RSNA_ASNR_MICCAI_BraTS2021_ValidationData.zip`
* Rename top level folder to `train`

## Train Model

Script for training a model.

Run the `train_model.py` script, selecting one of the parameters in the list for each argument.

```sh
python train_model.py --model [unet,segresnet] --epochs [1..n] --image_key [t1,t1ce,t2,flair]
```

The model with the best validation score will be saved to `trained_models/<model name>-model.pth`. Script will print
test validation score, and results can be viewd via tensorboard by running `tensorboard --logdir runs` or `make results`.

## Tune Hyperparameters

Script for running an [Optuna](https://optuna.org/) study for determining ideal model hyperparameters. Hyperparameters being tested can be modified within the `nn.optunet.Optunet` class.

Run the `tune_hyperparameters.py` script, selecting one of the parameters in the list for each argument.

```sh
python tune_hyperparameters.py -m [unet,segresnet] -e [1..n] --trials [1..n]
```

Note that the script can be stopped at any time with Ctrl+C and results will be retained in MLflow. To view the results
of the study in MLflow, run `mlflow server` or `make optuna_results`.

## Run Model

Script for running a trained model on an MRI.

Run the `tune_hyperparameters.py` script, specifying the following arguments.

```sh
python run_model.py -m <path to model> -i <path to NIfTI file of single channel MRI image> -l <path to NIfTI file of segmentation of MRI image>
```

The script will generate a MatPlotLib image including three slices of the input image, the ground truth segmentation,
and the segmentation generated by the model.

