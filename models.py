import numpy as np


def softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def cross_entropy_with_grad(proba, target):
    target_proba = np.choose(target, proba.T)
    target_proba += 1e-15 * np.ones_like(target_proba)  # overflow prevention
    loss = -np.log(target_proba).mean()

    grad = proba
    grad[range(grad.shape[0]), target] -= 1
    grad /= grad.shape[0]

    return loss, grad


class Classifier:
    def __init__(self, layers):
        self.layers = layers

    def parameters(self):
        return [layer_parameter for layer in self.layers for layer_parameter in layer.parameters()]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def __call__(self, x):
        return softmax(self.forward(x))
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train = False
