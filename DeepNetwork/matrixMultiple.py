from matrixToArray import *


def calculate_matrix_multiple(a, b):

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    deep_network = windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")

    matrix_multiply = deep_network.matrixMultiply
    matrix_multiply.argtypes = [matrix, matrix, matrix]

    A = getMatrixFromArray(a)
    B = getMatrixFromArray(b)

    c_count = A.rows * B.cols
    c_values = (c_float * c_count)()

    C = matrix(cast(c_values, POINTER(c_float)), A.rows, B.cols)

    matrix_multiply(A, B, C)

    return np.ctypeslib.as_array(c_values, shape=(c_count,))
