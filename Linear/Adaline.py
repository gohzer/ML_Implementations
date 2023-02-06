import numba
import numpy as np
from MLUtils.ModelBase import SupervisedModel
from MLUtils.WeightInitializers import random_normal, zero
from MLUtils.ActivationFunctions import heaviside
from MLUtils.LossFunctions import cost_function, cost_function_derivative

class Adaline(SupervisedModel):
    def __init__(self, x: np.ndarray, y: np.ndarray,
                 loss_function: callable, learning_rate: np.float,
                 weight_init: callable = random_normal):
        assert len(x) > 0
        assert len(x) == len(y)

        super().__init__(x, y, loss_function, learning_rate)

        self.one = np.array([1])
        self.length, self.width = self.x.shape
        self.weights = weight_init(self.width)

    def fit(self):
       self.weights = _adaline_fit(self.weights, self.x, self.y, self.lr)

    def infer_single(self, x: np.ndarray):
        return heaviside(np.dot(self.weights, np.hstack([self.one, x])))


@numba.njit
def _adaline_fit(weights, x, y, lr):
    for i in numba.prange(len(weights)):
        z = _adaline_infer_train(weights, x)
        negative_gradient = -cost_function_derivative(x[:, i], y, z)
        weights[i] += (negative_gradient * lr)
    return weights


@numba.njit(parallel=True)
def _adaline_infer_train(weights: np.ndarray, x: np.ndarray):
    z = np.zeros(len(x))
    for i in numba.prange(len(x)):
        z[i] = heaviside(np.dot(weights, x[i]))
    return z