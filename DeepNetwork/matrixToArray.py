from matrix import *
from ctypes import *
import numpy as np


def getMatrixFromArray(array):
    """
    Convert a numpy array into a matrix
    :param array:
    :return: matrix
    """

    array_shape = array.shape

    rows = array_shape[0]

    cols = 0
    if array_shape.__len__() == 1:
        cols = 1
    else:
        cols = array_shape[1]

    flattened_array = np.ravel(array)
    float_array = flattened_array.ctypes.data_as(POINTER(c_float))
    return matrix(float_array, rows, cols)