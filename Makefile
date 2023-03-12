PYTHON=python3

.PHONY: help clean install tar format results optuna_results unet

help:	               ## Show this help message
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

clean:	               ## Clean the directory
	git clean -dxf -e local_data/ -e trained_models/ -e .idea/ -e .vscode/ -e .python-version -e mlruns

install:               ## Install dependencies via pip
	pip install -r requirements.txt

tar:	               ## Tar the current project
	tar --exclude="./.git" --exclude="./__pycache" --exclude="local_data" --exclude="runs" --exclude="mlruns" --exclude="trained_models" -czvf project.tar .

format:	               ## Format python files in place with black formatter
	black -l 120 .

results:               ## See Neural Net results via Tensorboard
	tensorboard --logdir runs

optuna_results:        ## See Optuna results via MLflow
	mlflow server

unet:                  ## Train UNet model
	./train_model.py -m unet