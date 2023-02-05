import numba
import numpy as np
from MLUtils.ModelBase import SupervisedModel
from MLUtils.WeightInitializers import random_normal
from MLUtils.ActivationFunctions import heaviside


class Perceptron(SupervisedModel):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 loss_function: callable, learning_rate: np.float = .2):
        assert len(x) > 0
        assert len(x) == len(y)

        super().__init__(x, y, loss_function, learning_rate)

        self.one = np.array([1])
        self.length, self.width = self.x.shape
        self.weights = random_normal(self.width)

    def fit(self, epsilon=.01):
        diff = 1000
        while np.amax(diff) >= epsilon:
            w = _perceptron_fit(self.weights, self.x, self.y, self.loss_fn, self.lr)
            diff = np.abs(w - self.weights)
            self.weights = w

    def infer_single(self, x: np.ndarray):
        return heaviside(np.dot(self.weights, np.hstack([self.one, x])))


@numba.njit
def _perceptron_fit(weights, x, y, loss_fn, lr):
    height = len(x)
    for i in numba.prange(height):
        y_bar = _perceptron_infer_train(weights, x[i])
        loss = np.full_like(x[i], loss_fn(y[i], y_bar) * lr)
        deltas = np.multiply(loss, x[i])
        weights += deltas
    return weights


@numba.njit
def _perceptron_infer_train(weights: np.ndarray, x: np.ndarray, activation: callable = heaviside):
    res = activation(np.dot(weights, x))
    return res