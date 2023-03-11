from copy import deepcopy

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.visualize import plot_2d_or_3d_image
import torch
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


class NNet:
    def __init__(self, model, optimizer, alpha, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = ""
        self.model = model.to(self.device)
        self.lf = DiceCELoss(sigmoid=True)
        try:
            self.optim = optimizer(self.model.parameters(), alpha)
        except TypeError:
            self.optim = optimizer(self.model.parameters())
        self.scheduler = ExponentialLR(self.optim, gamma)
        self.best_model_weights = None

    def run_training(
        self,
        train_dataloader,
        val_dataloader,
        postproc_func,
        val_metric,
        epochs,
        summary_writer=None,
    ):
        best_metric = -torch.inf
        for epoch in tqdm(range(epochs), desc=f"Training over {epochs} epochs"):
            self._train(train_dataloader, epoch, summary_writer)
            best_metric = self._validation(
                val_dataloader,
                postproc_func,
                val_metric,
                best_metric,
                epoch,
                summary_writer,
            )
        return best_metric

    def save_model(self, filepath):
        torch.save(self.best_model_weights, filepath)

    def _train(self, dataloader, epoch=None, summary_writer=None):
        self.model.train()
        running_loss = 0
        for batch in dataloader:
            image, label = batch["flair"].to(self.device), batch["seg"].to(self.device)
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(image)
                loss = self.lf(outputs, label)
                loss.backward()
                self.optim.step()
            running_loss += loss.item()
        total_loss = running_loss / (len(dataloader.dataset))
        if summary_writer and epoch is not None:
            summary_writer.add_scalar("training_loss", total_loss, epoch)
        self.scheduler.step()
        if epoch is not None:
            print(f"Epoch {epoch} Training Loss: {total_loss:.4f}")
            print("-" * 25)

    def _validation(
        self,
        dataloader,
        postproc_func,
        val_metric,
        best_metric,
        epoch=None,
        summary_writer=None,
    ):
        self.model.eval()
        image, label, output = None, None, None
        with torch.no_grad():
            for batch in dataloader:
                image, label = batch["flair"].to(self.device), batch["seg"].to(
                    self.device
                )  # torch.Size([3, 1, 128, 128, 64])
                roi_size = (96, 96, 96)
                output = sliding_window_inference(image, roi_size, 4, self.model)
                output = torch.stack(
                    [postproc_func(i) for i in decollate_batch(output)]
                )
                binarized_y = [postproc_func(i) for i in decollate_batch(label)]
                val_metric(y_pred=output, y=binarized_y)
            metric = val_metric.aggregate().item()
            val_metric.reset()
            if epoch is not None:
                print(f"Epoch {epoch} Mean Dice: {metric:.4f}")
                print("-" * 25)
            if metric > best_metric:
                best_metric = metric
                self.best_model_weights = deepcopy(self.model.state_dict())
                if summary_writer and epoch is not None:
                    plot_2d_or_3d_image(
                        image, epoch + 1, summary_writer, index=0, tag="image"
                    )
                    plot_2d_or_3d_image(
                        output, epoch + 1, summary_writer, index=0, tag="label"
                    )
                    plot_2d_or_3d_image(
                        label, epoch + 1, summary_writer, index=0, tag="true_label"
                    )

            if summary_writer:
                summary_writer.add_scalar("validation_mean_dice", metric, epoch)
        return best_metric
