from src.utils import EnergyFunction
import torch.nn as nn

class MSEEnergy(EnergyFunction):
    def __init__(self):
        self.loss = nn.MSELoss(reduction='sum')

    def calculate(self, inputs, targets):
        return self.loss(inputs, targets)

class BCEEnergy(EnergyFunction):
    def __init__(self, logit=False):
        if logit:
            self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        else:
            self.loss = nn.BCELoss(reduction='sum')

    def calculate(self, inputs, targets):
        return self.loss(inputs, targets)