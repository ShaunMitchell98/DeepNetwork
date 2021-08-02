import ctypes
from matrix import matrix
import numpy as np


def forward_propagate_layer(A, B):

    deep_network = ctypes.windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")

    deep_network_forward_propagate_layer = deep_network.forward_propagate_layer
    deep_network_forward_propagate_layer.argtypes = [matrix, matrix, matrix, ctypes.c_int]

    c_count = A.rows * B.cols
    c_values = (ctypes.c_double * c_count)()

    C = matrix(ctypes.cast(c_values, ctypes.POINTER(ctypes.c_double)), A.rows, B.cols)

    value = np.ctypeslib.as_array(A.values, shape=(A.rows * A.cols,))
    deep_network_forward_propagate_layer(A, B, C, 0)

    return C