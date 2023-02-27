BATCH_SIZE = 4
WORKERS = 4
EPOCHS = 10
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
VAL_RATIO = 0.15
CPU = "cpu"
GPU = "cuda"
LOCAL_DATA = {
    "train": "local_data/train",
    "validation": "local_data/validation",
    "cache": "local_data/persistent_dataset",
}
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
}