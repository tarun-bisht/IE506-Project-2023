from abc import ABC, abstractmethod

class EnergyFunction(ABC):

    @abstractmethod
    def calculate(self, inputs, target):
        pass