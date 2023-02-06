import numba
import numpy as np
from MLUtils.ModelBase import SupervisedModel
from MLUtils.WeightInitializers import random_normal, zero
from MLUtils.ActivationFunctions import heaviside, relu
from MLUtils.LossFunctions import cost_function, cost_function_derivative

class Adaline(SupervisedModel):
    def __init__(self, learning_rate: np.float,
                 weight_init: callable = random_normal):

        super().__init__(learning_rate)
        self.weights = None
        self.bias = 0.0

    def fit(self, x, y, epochs=100, tol=1e-8):
        pass

    def predict(self, x):
        pass


