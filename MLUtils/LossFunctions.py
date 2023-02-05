import numba
import numpy as np

@numba.njit(fastmath=True)
def implicit_loss(y, y_bar):
    return y - y_bar