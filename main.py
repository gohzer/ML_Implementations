from Linear.Perceptron import Perceptron
from MLUtils.LossFunctions import implicit_loss
import numpy as np
from sklearn import datasets
import timeit
import matplotlib.pyplot as plt

# --> Import sklearn utility functions to create derived-class objects.
from sklearn.base import BaseEstimator, ClassifierMixin


# --> Redefine the Heaviside function.
def H(x): return np.heaviside(x - 0.5, 1).astype(np.int)


class Adaline(BaseEstimator, ClassifierMixin):
    """
    Implementation of Adaline using sklearn BaseEstimator and
    ClassifierMixin.
    """

    def __init__(self, learning_rate=0.001, epochs=100, tol=1e-8):

        # --> Learning rate for the delta rule.
        self.learning_rate = learning_rate

        # --> Maximum number of epochs for the optimizer.
        self.epochs = epochs

        # --> Tolerance for the optimizer.
        self.tol = tol

    def predict(self, X):
        return H(self.weighted_sum(X))

    def weighted_sum(self, X):
        return X @ self.weights + self.bias

    def fit(self, X, y):
        """
        Implementation of the Delta rule for training Adaline.
        INPUT
        -----
        X : numpy 2D array. Each row corresponds to one training example.
        y : numpy 1D array. Label (0 or 1) of each example.
        OUTPUT
        ------
        self: The trained adaline model.
        """

        # --> Number of features.
        n = X.shape[1]

        # --> Initialize the weights and bias.
        self.weights = np.zeros((n,))
        self.bias = 0.0

        # --> Training of Adaline using the Delta rule.
        for _ in range(self.epochs):

            # --> Compute the error.
            error = self.weighted_sum(X) - y

            # --> Update the weights and bias.
            self.weights -= self.learning_rate * error @ X
            self.bias -= self.learning_rate * error.sum()

            # --> Check for convergence.
            if np.linalg.norm(error) < self.tol:
                break

        return self

def main():
    # test dataset
    dataset_size = 200
    test_split = .2
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    X, y = datasets.make_classification(
        n_features=2,
        n_classes=2,
        n_samples=dataset_size,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=7
    )

    # plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
    # plt.show()

    trainx = X[:train_size]
    trainy = y[:train_size]

    testx = X[train_size:]
    testy = y[train_size:]

    model = Perceptron(learning_rate=0.001)
    model.fit(trainx, trainy)
    y_pred = model.predict(testx)
    print(y_pred)
    correct = (y_pred == testy).sum()
    print(f"Acc: {correct / test_size * 100: .2f}")


if __name__ == '__main__':
    main()
