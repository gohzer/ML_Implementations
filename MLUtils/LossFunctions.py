import numba
import numpy as np

@numba.njit
def implicit_loss(y, y_bar):
    return y - y_bar