import numpy as np


class MomentumSGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0.0] * len(parameters)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.value)

    def step(self):
        for i, parameter in enumerate(self.parameters):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * parameter.grad
            parameter.value += self.velocities[i]
