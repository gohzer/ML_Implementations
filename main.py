from Linear import Perceptron
from MLUtils.LossFunctions import implicit_loss
import numpy as np

def main():
    # test nand
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 1, 1, 0])

    model = Perceptron.Perceptron(x, y, implicit_loss, learning_rate=0.2)
    for _ in range(100):
        model.fit()

    for i in x:
        y = model.infer_single(i)
        print(i, y)

if __name__ == '__main__':
    main()
