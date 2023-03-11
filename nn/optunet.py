from torch import optim

from .nnet import NNet


class Optunet(NNet):
    def __init__(self, trial, model):
        self.trial = trial
        optimizer = self.select_optimizer()
        alpha = trial.suggest_float("lr", 5e-7, 5e-3, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1e-3, log=True)
        super().__init__(model, optimizer, alpha, gamma)

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
