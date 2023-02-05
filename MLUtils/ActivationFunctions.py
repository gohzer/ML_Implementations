import numba
import numpy

def heaviside(z):
    if z >= 0:
        return 1
    else:
        return 0

def heaviside_height_2(z):
    if z >= 0:
        return 1
    else:
        return -1