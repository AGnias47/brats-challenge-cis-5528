from torch import optim

from .nnet import NNet


class Optunet(NNet):
    def __init__(self, trial, model):
        self.trial = trial
        optimizer = self.select_optimizer()
        alpha = trial.suggest_float("lr", 5e-4, 5e-1, log=True)
        super().__init__(model, optimizer, alpha)

    def select_optimizer(self):
        optimizer_name = self.trial.suggest_categorical(
            "optimizer",
            [
                "ASGD",
                "Adadelta",
                "Adagrad",
                "Adam",
                "AdamW",
                "Adamax",
                "NAdam",
                "RAdam",
                "RMSprop",
            ],
        )
        return getattr(optim, optimizer_name)
