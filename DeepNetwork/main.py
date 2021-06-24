from ctypes import *
import numpy as np


def calculate_matrix_multiple(a, b):

    deep_network = windll.LoadLibrary(r"C:\Users\Shaun Mitchell\source\repos\DeepNetwork\x64\Release\DeepNetwork.Infrastructure.dll")
    matrix_multiply = deep_network.matrixMultiply
    matrix_multiply.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int]
    c = (c_float * len(a))()

    matrix_multiply(a.ctypes.data_as(POINTER(c_float)), b.ctypes.data_as(POINTER(c_float)), cast(c, POINTER(c_float)),
                    c_int(int(len(a)/2)))

    return np.ctypeslib.as_array(c, shape=(len(a),))


a = np.array([3, 3, 3, 3]).astype(np.float32)
b = np.array([2, 2, 2, 2]).astype(np.float32)
result = calculate_matrix_multiple(a, b)
e = 5