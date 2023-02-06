import numpy as np

def random_normal(len):
    w = np.random.normal(loc=0.0, scale=0.01, size=len)
    return w

def zero(len):
    return np.zeros(len)