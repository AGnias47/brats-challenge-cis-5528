from copy import deepcopy
from functools import wraps
import logging
from time import time

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
import torch
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from config import IMAGE_RESOLUTION, LOCAL_DATA, IMAGE_KEY, LABEL_KEY
from data.transforms import validation_postprocessor


logger = logging.getLogger(__name__)


def timing(f):
    """
    Taken from https://stackoverflow.com/a/27737385/8728749
    Parameters
    ----------
    f

    Returns
    -------

    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info("[%r] %2.4fs" % (f.__name__, te - ts))
        return result

    return wrap


class NNet:
    def __init__(self, name, model, optimizer, alpha, gamma=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.model = model.to(self.device)
        self.lf = DiceCELoss(sigmoid=True)
        self.postproc_func = validation_postprocessor()
        self.val_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        try:
            self.optim = optimizer(self.model.parameters(), alpha)
        except TypeError:
            self.optim = optimizer(self.model.parameters())
        if gamma:
            self.scheduler = ExponentialLR(self.optim, gamma)
        else:
            self.scheduler = None

    @timing
    def run_training(
        self,
        train_dataloader,
        val_dataloader,
        epochs,
        summary_writer=None,
    ):
        best_metric = -torch.inf
        for epoch in tqdm(range(epochs), desc=f"Training over {epochs} epochs"):
            self._train(train_dataloader, epoch, summary_writer)
            best_metric = self._validation(
                val_dataloader,
                best_metric,
                epoch,
                summary_writer,
            )
        return best_metric

    def save_model(self):
        best_model_weights = deepcopy(self.model.state_dict())
        torch.save(best_model_weights, f"{LOCAL_DATA['model_output']}/{self.name}-model.pth")

    def load_model(self):
        self.model.load_state_dict(torch.load(f"{LOCAL_DATA['model_output']}/{self.name}-model.pth"))

    @timing
    def _train(self, dataloader, epoch=None, summary_writer=None):
        self.model.train()
        running_loss = 0
        for batch in dataloader:
            image, label = batch[IMAGE_KEY].to(self.device), batch[LABEL_KEY].to(self.device)
            self.optim.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(image)
                loss = self.lf(outputs, label)
                loss.backward()
                self.optim.step()
            running_loss += loss.item()
        total_loss = running_loss / len(dataloader)
        if summary_writer and epoch is not None:
            summary_writer.add_scalar("training_loss", total_loss, epoch)
        if self.scheduler:
            self.scheduler.step()
        if epoch is not None:
            print(f"Epoch {epoch} Training Loss: {total_loss:.4f}")
            print("-" * 25)

    @timing
    def _validation(
        self,
        dataloader,
        best_metric,
        epoch=None,
        summary_writer=None,
    ):
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                image, label = batch[IMAGE_KEY].to(self.device), batch[LABEL_KEY].to(self.device)
                self.optim.zero_grad()
                output = sliding_window_inference(
                    inputs=image,
                    roi_size=IMAGE_RESOLUTION,
                    sw_batch_size=image.size()[0],
                    predictor=self.model,
                    overlap=0.5,
                )
                output = torch.stack([self.postproc_func(i) for i in decollate_batch(output)])
                self.val_metric(y_pred=output, y=label)
            metric = self.val_metric.aggregate().item()
            self.val_metric.reset()
            if epoch is not None:
                print(f"Epoch {epoch} Mean Dice: {metric:.4f}")
                print("-" * 25)
            if metric > best_metric:
                best_metric = metric
                self.save_model()
            if summary_writer:
                summary_writer.add_scalar("validation_mean_dice", metric, epoch)
        return best_metric

    @timing
    def test(
        self,
        dataloader,
        summary_writer=None,
    ):
        self.load_model()
        self.model.eval()
        image, label, output = None, None, None
        with torch.no_grad():
            for batch in dataloader:
                image, label = batch[IMAGE_KEY].to(self.device), batch[LABEL_KEY].to(self.device)
                self.optim.zero_grad()
                output = sliding_window_inference(
                    inputs=image,
                    roi_size=IMAGE_RESOLUTION,
                    sw_batch_size=image.size()[0],
                    predictor=self.model,
                    overlap=0.5,
                )
                output = torch.stack([self.postproc_func(i) for i in decollate_batch(output)])
                self.val_metric(y_pred=output, y=label)
            metric = self.val_metric.aggregate().item()
            self.val_metric.reset()
            if summary_writer:
                plot_2d_or_3d_image(image, 1, summary_writer, index=0, tag=IMAGE_KEY)
                plot_2d_or_3d_image(output, 1, summary_writer, index=0, tag="label")
                plot_2d_or_3d_image(label, 1, summary_writer, index=0, tag="true_label")
        return metric
