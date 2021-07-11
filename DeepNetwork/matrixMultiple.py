from matrixToArray import *


def calculate_matrix_multiple(A, B):

    deep_network = windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")

    matrix_multiply = deep_network.matrix_multiply
    matrix_multiply.argtypes = [matrix, matrix, matrix]

    c_count = A.rows * B.cols
    c_values = (c_float * c_count)()

    C = matrix(cast(c_values, POINTER(c_float)), A.rows, B.cols)

    matrix_multiply(A, B, C)

    return C
