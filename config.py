from monai.data import list_data_collate

# Training parameters
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
VAL_RATIO = 0.15

# Dataset parameters
PERSIST_DATASET = False
IMAGE_RESOLUTION = (128, 128, 64)

# Dataloader parameters
BATCH_SIZE = 1
WORKERS = 1
DATALOADER_KWARGS_CPU = {
    "batch_size": BATCH_SIZE,
    "num_workers": WORKERS,
    "shuffle": True,
}
DATALOADER_KWARGS_GPU = {
    "batch_size": BATCH_SIZE,
    "num_workers": WORKERS,
    "pin_memory": True,
    "shuffle": True,
    "collate_fn": list_data_collate,
}

# Filepath parameters
LOCAL_DATA = {
    "train": "local_data/train_subset",
    "validation": "local_data/validation",
    "cache": "local_data/persistent_dataset",
    "model_output": "trained_models",
    "tensorboard_logs": "runs",
}
