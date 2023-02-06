import numba
import numpy as np

@numba.njit
def implicit_loss(y, y_bar):
    return y - y_bar

@numba.njit(fastmath=True)
def cost_function(y, z):
    return (1/2) * np.sum((y - z) ** 2)

@numba.njit(fastmath=True)
def cost_function_derivative(x, y, z):
    return -1 * np.sum(np.multiply((y - z), x))