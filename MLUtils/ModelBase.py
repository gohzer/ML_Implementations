import numpy as np

class Model:
    def __init__(self, loss_function, learning_rate):
        self.loss_fn = loss_function
        self.lr = learning_rate


class SupervisedModel(Model):
    def __init__(self, x, y, loss_function, learning_rate):
        super().__init__(loss_function, learning_rate)
        self.ones = np.ones(len(x))
        self.x = np.vstack([self.ones, x.T]).T
        self.y = y