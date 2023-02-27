from copy import deepcopy

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
import torch
import torch.optim
from tqdm import tqdm


def train(
    device,
    model,
    loss_function,
    optimizer,
    scheduler,
    validation_metric,
    validation_postprocessor,
    train_dataloader,
    validation_dataloader,
    epochs=10,
):
    best_model_wts = deepcopy(model.state_dict())
    best_metric = -torch.inf
    for epoch in tqdm(range(epochs), desc=f"Training over {epochs} epochs"):
        training_step(
            device,
            model,
            loss_function,
            optimizer,
            scheduler,
            train_dataloader,
            epoch,
        )
        validation_step(
            device,
            model,
            validation_metric,
            validation_postprocessor,
            validation_dataloader,
            best_metric,
            epoch
        )
    model.load_state_dict(best_model_wts)
    return model


def training_step(
    device,
    model,
    loss_function,
    optimizer,
    scheduler,
    train_dataloader,
    epoch,
):
    running_loss = float(0)
    model.train()
    for batch in train_dataloader:
        image, label = batch["flair"].to(device), batch["seg"].to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(image)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * image.size(0)
    total_loss = running_loss / (len(train_dataloader.dataset))
    scheduler.step()
    print(f"Epoch {epoch} Training Loss: {total_loss:.4f}")
    print("-" * 25)


def validation_step(
            device,
            model,
            validation_metric,
            validation_postprocessor,
            validation_dataloader,
            best_metric,
            epoch
        ):
    model.eval()
    with torch.no_grad():
        for batch in validation_dataloader:
            image, label = batch["flair"].to(device), batch["seg"].to(device)
            roi_size = (96, 96, 96)
            output = sliding_window_inference(image, roi_size, 4, model)
            output = [validation_postprocessor(i) for i in decollate_batch(output)]
            metric = validation_metric(y_pred=output, y=label)
        metric_result = metric.aggregate.item()
        metric.reset()
        print(f"Epoch {epoch} Mean Dice: {metric_result:.4f}")
        print("-" * 25)
        best_model_wts = None
        if metric > best_metric:
            best_metric = metric
            best_model_wts = deepcopy(model.state_dict())
        return best_metric, best_model_wts
