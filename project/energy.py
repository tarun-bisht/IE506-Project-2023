from src.utils import EnergyFunction
import torch.nn as nn

class MSEEnergy(EnergyFunction):
    def __init__(self):
        self.loss = nn.MSELoss()

    def calculate(self, inputs, targets):
        return self.loss(inputs, targets)

class BCEEnergy(EnergyFunction):
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss()

    def calculate(self, inputs, targets):
        return self.loss(inputs, targets)