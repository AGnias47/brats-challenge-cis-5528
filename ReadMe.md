# BraTS Tumor Segmentation Pipeline

This project contains the Tumor Segmentation Pipeline used to train a model for brain tumor segmentation using data from various [Synapse BraTS challenges](https://www.synapse.org/Synapse:syn53708126/wiki/626320). This pipeline was initially created for a group project in Temple University's CIS 5528, Predictive Modeling in Biomedicine. The final report for this project can be found [here](results/report.pdf).

## Data

For this class, data from the [BraTS Continuous Evaluation](https://www.synapse.org/Synapse:syn27046444/wiki/616571) competition was used. The original data used is no longer available, but the competition is now part of [BraTS 2023 - Adult Glioma](https://www.synapse.org/Synapse:syn51156910/wiki/622351). Data from that competition is available and supported by this repo.

## Implementation

The pipeline heavily uses the [MONAI](https://monai.io/) module, which uses a [PyTorch](https://pytorch.org/) backend. The pipeline can be configured to train either MONAI's built-in UNet model or a SegResNet model. The pipeline runs a test loop with the best model weights found during training. Models are saved in the `trained_models` directory, and can be used in the [run_model.py](run_model.py) script.

## Ongoing Support

Currently, there are no plans to support future iterations of the BraTS competition. The update supporting 2023 data was mainly done because the original training data was lost and could not be recovered through Synapse. That being said, updates needed between competition years should be minimal. Commit [7a96658b](https://github.com/AGnias47/brats-challenge-cis-5528/commit/7a96658b25ee07bff311ad57a6a91b7253f330e4) shows the changes that were necessary for `config.py` and `data/containers.py` to support the updated dataset.

## Local Setup

### Hardware

A GPU is required in order to train a model in a reasonable timeframe. When using a GeForce RTX 3060 Ti 8GB GPU, each
training epoch took ~20 minutes, so training over 150 epochs took over 2 days.

### Python Setup

The repo was built with Python 3.9 and currently supports Python 3.12. Dependencies can be installed via the requirements file, though it is recommended PyTorch be installed first following [PyTorch's installation instructions](https://pytorch.org/get-started/locally/).

### Download Data

* Download zip of training data from the BraTS competition
    * Request access to competition data from Synapse [here](https://www.synapse.org/Synapse:syn51156910/wiki/627000) and download the training data zip directly from their site.
* Move the file to `./local_data`
* Unzip the file
  * `unzip ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`
* Rename top level folder to `train`

## Train Model

Script for training a model.

```sh
python train_model.py --model [unet,segresnet] --epochs [1..n]
```

By default, `unet` and `150` are used.

### View Training Results

The model with the best validation score will be saved to `trained_models/<model name>-model.pth`. Script will print
test validation score, and results can be viewed via TensorBoard by running `tensorboard --logdir runs` or `make results`.

Training Loss and Validation Score data can also be exported from TensorBoard. If exported in a `*.json`, it can be used
in the [generate_graphs.py](results/generate_graphs.py) script to generate a plot of the loss and validation over each
epoch. Note that this script is not paramaterized, so paths, etc. will need to be updated within the script, or set to
match its current contents.

## Tune Hyperparameters

Script for running an [Optuna](https://optuna.org/) study for determining ideal model hyperparameters. Hyperparameters
being tested can be modified within the `nn.optunet.Optunet` class.

Run the [tune_hyperparameters.py](tune_hyperparameters.py) script, selecting one of the parameters in the list for each argument.

```sh
python tune_hyperparameters.py -m [unet,segresnet] -e [1..n] --trials [1..n]
```

A model type must be selected, but by default, 50 epochs and 20 trials are used.

Note that the script can be stopped at any time with Ctrl+C and results will be retained in MLflow.

### View Hyperparameter Optimization Results

To view the results of the study in MLflow, run `mlflow server` or `make optuna_results`.

## Run Model

Script for running a trained model on an MRI.

Run the [run_model.py](run_model.py) script, specifying the following arguments.

```sh
python run_model.py -m <path to model> -i <path to NIfTI subdirectory containing all channel images and segmentation>
```

The script will generate a MatPlotLib image including three slices of the input image, the ground truth segmentation,
and the segmentation generated by the model.
