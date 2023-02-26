#!/usr/bin/env python3

"""
Andy Gnias
Temple University
Center for Data Analytics and Biomedical Informatics

helpers.py
--------
Generic utilities used in this project
"""

import logging
import os
import random
import threading
from time import sleep

import numpy as np
from psutil import virtual_memory
import torch
import torch.backends.cudnn


def seed_everything(seed=42):
    """
    Seed everything.

    Source
    ------
    https://github.com/AGnias47/russo-ukranian-tweet-classification/blob/main/utils/utils.py

    Parameters
    ----------
    seed: int (default is 42)

    Returns
    -------
    None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return None


def log_memory(interval=60):
    """
    Log memory usage every x seconds defined by an interval. Set thread attribute `run` to False to stop.

    Parameters
    ----------
    interval: int (default is 60)
        Seconds between logging

    Sources
    -------
    https://stackoverflow.com/questions/60994719/python-break-process-if-memory-consumption-is-to-high
    https://stackoverflow.com/questions/18018033/how-to-stop-a-looping-thread-in-python

    Returns
    -------
    None
    """
    t = threading.current_thread()
    while getattr(t, "run", True):
        logging.info("Memory usage: %i%%", int(virtual_memory().percent))
        sleep(interval)
