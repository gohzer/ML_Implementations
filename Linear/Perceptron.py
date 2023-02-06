import numba
import numpy as np
from MLUtils.ModelBase import SupervisedModel
from MLUtils.WeightInitializers import random_normal
from MLUtils.ActivationFunctions import heaviside


class Perceptron(SupervisedModel):
    def __init__(self, learning_rate: np.float = .2, weight_init=random_normal):
        super().__init__(learning_rate, weight_init)

    def fit(self, x, y, epochs=200, tol=0):
        self.weights = self.init(len(x[0])+1)
        error = np.inf
        for _ in range(epochs):
            for yi, xi in zip(y, x):
                pred = self.predict(xi)
                error = self.lr * (yi - pred)
                deltas = np.multiply(xi, error)
                self.weights[1:] += deltas
                self.weights[0] += error
            if error < tol:
                break

    def predict(self, x):
        return np.heaviside(x @ self.weights[1:] + self.weights[0], 1)
