PYTHON=python3

.PHONY: help clean tar format results unet

help:	      ## Show this help message
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

clean:	      ## Clean the directory
	git clean -dxf -e local_data/ -e model-output/ -e .idea/ -e .vscode/ -e .python-version -e mlruns

tar:	      ## Tar the current project
	tar --exclude="./.git" --exclude="./__pycache" -czvf project.tar .

format:	      ## Format python files in place with black formatter
	black -l 120 .

results:      ## See Neural Net results via Tensorboard
	tensorboard --logdir runs

unet:         ## Train UNet model
	./train_model.py -m unet