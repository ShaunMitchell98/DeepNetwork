from ctypes import *
import numpy as np
from matrix import matrix

def calculate_matrix_multiple(A, B):

    deep_network = windll.LoadLibrary(r"C:\Users\Shaun Mitchell\source\repos\DeepNetwork\x64\Release\DeepNetwork.Infrastructure.dll")

    cCount = A.rows * B.cols
    matrix_multiply = deep_network.matrixMultiply
    matrix_multiply.argtypes = [matrix, matrix, matrix]
    cValues = (c_float * cCount)()
    C = matrix(cast(cValues, POINTER(c_float)), B.cols, A.rows)

    matrix_multiply(A, B, C)

    return np.ctypeslib.as_array(cValues, shape=(cCount,))