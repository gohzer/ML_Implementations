from Linear import Perceptron
from MLUtils.LossFunctions import implicit_loss
import numpy as np
from sklearn import datasets
import timeit
import matplotlib.pyplot as plt

def main():
    # test dataset
    dataset_size = 5000
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

    plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
    plt.show()

    trainx = X[:train_size]
    trainy = y[:train_size]

    testx = X[train_size:]
    testy = y[train_size:]

    start = timeit.default_timer()
    model = Perceptron.Perceptron(trainx, trainy, implicit_loss, learning_rate=0.05)
    model.fit()

    correct = 0
    for x, y in zip(testx, testy):
        y_bar = model.infer_single(x)
        if y == y_bar:
            correct += 1

    print(f"Acc: {correct/test_size * 100: .2f}%")

    end = timeit.default_timer()
    print(end-start)

if __name__ == '__main__':
    main()
