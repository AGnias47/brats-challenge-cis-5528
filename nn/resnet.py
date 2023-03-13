from torch import optim
from torchvision.models import resnet50
from .nnet import NNet


class ResNet(NNet):
    def __init__(self):
        self.name = "resnet"
        model = resnet50()
        optimizer = optim.Adam
        alpha = 1e-2
        super().__init__(model, optimizer, alpha)
