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

    def fit(self):
        for i in range(self.length):
            x = self.x[i]
            y = self.y[i]
            y_bar = self.infer_train(x)
            loss = self.loss_fn(y, y_bar) * self.lr
            for j in range(self.width):
                delta_w = loss * x[j]
                self.weights[j] += delta_w

    def infer_train(self, x: np.ndarray, activation=heaviside):
        return activation(np.dot(self.weights, x))

    def infer_single(self, x: np.ndarray, activation=heaviside):
        return activation(np.dot(self.weights, np.hstack([self.one, x])))