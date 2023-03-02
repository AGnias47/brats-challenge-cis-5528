from copy import deepcopy

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.visualize import plot_2d_or_3d_image
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
    summary_writer,
    epochs=10,
):
    best_metric = -torch.inf
    for epoch in tqdm(range(epochs), desc=f"Training over {epochs} epochs"):
        for step in ["training", "validation"]:
            if step == "training":
                model.train()
                running_loss = 0
                for batch in train_dataloader:
                    image, label = batch["flair"].to(device), batch["seg"].to(device)
                    optimizer.zero_grad()
                    outputs = model(image)
                    loss = loss_function(outputs, label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                total_loss = running_loss / (len(train_dataloader.dataset))
                if summary_writer:
                    summary_writer.add_scalar("training_loss", total_loss, epoch)
                scheduler.step()
                print(f"Epoch {epoch} Training Loss: {total_loss:.4f}")
                print("-" * 25)
            elif epoch % 2 != 0:
                model.eval()
                image, label, output = None, None, None
                with torch.no_grad():
                    for batch in validation_dataloader:
                        image, label = batch["flair"].to(device), batch["seg"].to(
                            device
                        )
                        roi_size = (96, 96, 96)
                        output = sliding_window_inference(image, roi_size, 4, model)
                        output = [
                            validation_postprocessor(i) for i in decollate_batch(output)
                        ]
                        validation_metric(y_pred=output, y=label)
                    metric = validation_metric.aggregate().item()
                    validation_metric.reset()
                    print(f"Epoch {epoch} Mean Dice: {metric:.4f}")
                    print("-" * 25)
                    if metric > best_metric:
                        best_metric = metric
                        best_model_wts = deepcopy(model.state_dict())
                        if summary_writer:
                            plot_2d_or_3d_image(
                                image, epoch + 1, summary_writer, index=0, tag="image"
                            )
                            plot_2d_or_3d_image(
                                label, epoch + 1, summary_writer, index=0, tag="label"
                            )
                            plot_2d_or_3d_image(
                                output, epoch + 1, summary_writer, index=0, tag="output"
                            )
                    if summary_writer:
                        summary_writer.add_scalar("validation_mean_dice", metric, epoch)
    return best_model_wts
