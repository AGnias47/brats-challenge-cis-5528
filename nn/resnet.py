from torch import optim
from torchvision.models import resnet50
from .nnet import NNet


class ResNet(NNet):
    def __init__(self):
        self.name = "resnet"
        model = resnet50()
        optimizer = optim.Adam
        alpha = 0.1
        gamma = 4.182657295694138e-05
        super().__init__(model, optimizer, alpha, gamma)
