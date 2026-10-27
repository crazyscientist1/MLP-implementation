import numpy as np
import math

class ReLU:
    activation_func = staticmethod(lambda x: np.maximum(x, 0))

    @staticmethod
    def backward(grad, z):
        return grad * (z > 0)

class Tanh:
    activation_func = staticmethod(np.tanh)

    @staticmethod
    def backward(grad, z):
        return grad * (1 - np.tanh(z) ** 2)

class Linear:
    activation_func = staticmethod(lambda x: x)

    @staticmethod
    def backward(grad, z):
        return grad

class Softmax:
    @staticmethod
    def activation_func(z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def backward(grad, z):
        S = Softmax.activation_func(z)
        return S * (grad - np.sum(grad * S, axis=1, keepdims=True))

class MSE:
    activation_func = staticmethod(lambda y, y_hat: sum((y - y_hat)**2)/y.shape[0])
    derivative_func = staticmethod(lambda y, y_hat: -2 * (y - y_hat)/y.shape[0])


class Cross_Entropy:
    @staticmethod
    def activation_func(y, y_hat):
        y_hat = np.clip(y_hat, 1e-12, 1.0)
        return -1 * np.sum(y * np.log(y_hat)) / y.shape[0]
    @staticmethod
    def derivative_func(y, y_hat):
        return -1 * y / y_hat / y.shape[0]
    