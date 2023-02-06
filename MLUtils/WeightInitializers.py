import numpy as np

def random_normal(len):
    rng = np.random.default_rng()
    weights = rng.standard_normal(len)
    return weights

def zero(len):
    return np.zeros(len)