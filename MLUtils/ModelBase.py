import numpy as np

class Model:
    def __init__(self, loss_function, learning_rate, weight_init):
        self.loss_fn = loss_function
        self.lr = learning_rate
        self.init = weight_init
        self.ONE = np.array([1])

    def predict(self, x):
        pass


class SupervisedModel(Model):
    def __init__(self, learning_rate, weight_init):
        super().__init__(None, learning_rate, weight_init)

    def fit(self, x, y, epochs, tol):
        pass