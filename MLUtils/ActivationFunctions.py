import numba
import numpy as np

@numba.njit
def heaviside(z) -> np.float:
    if z >= 0:
        return 1
    else:
        return 0

@numba.njit
def heaviside_height_2(z) -> np.float:
    if z >= 0:
        return 1
    else:
        return -1

@numba.njit
def relu(z):
    if z >= 0:
        return z
    else:
        return 0