import ctypes
from matrix import matrix


def forward_propagate_layer(A, B):

    deep_network = ctypes.windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")

    deep_network_forward_propagate_layer = deep_network.forward_propagate_layer
    deep_network_forward_propagate_layer.argtypes = [matrix, matrix, matrix, ctypes.c_int]

    c_count = A.rows * B.cols
    c_values = (ctypes.c_float * c_count)()

    C = matrix(ctypes.cast(c_values, ctypes.POINTER(ctypes.c_float)), A.rows, B.cols)

    deep_network_forward_propagate_layer(A, B, C, 0)

    return C