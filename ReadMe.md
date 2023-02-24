# CIS 5528 BraTS Project

Repository for all code related to project.

## Initial Setup

* Install pyenv via instructions at https://github.com/pyenv/pyenv-virtualenv and update .*rc file accordingly
* Install Python 3.9 `pyenv install 3.9.16`
* Create venv `pyenv virtualenv 3.9.16 brats`
* Activate venv `pyenv local brats`
* Install dependencies `pip install -r requirements.txt`

### Download Data

* Download zip from https://drive.google.com/file/d/1yJKfjNWYCv0hWQkX9wiE4HM1p9ButSKW/view?usp=share_link
* Move the file to `./local_data`
* Unzip `unzip RSNA_ASNR_MICCAI_BraTS2021_ValidationData.zip`
