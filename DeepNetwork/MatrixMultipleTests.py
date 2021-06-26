import unittest
import numpy as np
from matrix import *
from matrixMultiple import *


class MyTestCase(unittest.TestCase):
    def test_GivenSquareMatrix_ReturnsResult(self):

        a = np.array([3, 3, 3, 3]).astype(np.float32)
        b = np.array([2, 2, 2, 2]).astype(np.float32)
        A = matrix(a.ctypes.data_as(POINTER(c_float)), 2, 2)
        B = matrix(b.ctypes.data_as(POINTER(c_float)), 2, 2)

        c = calculate_matrix_multiple(A, B)

        np.testing.assert_array_equal([12, 12, 12, 12], c)

    def test_GivenNonSquareMatrix_ReturnsResult(self):

        a = np.array([1, 1, 1, 1, 1, 1]).astype(np.float32)
        b = np.array([1, 1, 1, 1, 1, 1]).astype(np.float32)
        A = matrix(a.ctypes.data_as(POINTER(c_float)), 2, 3)
        B = matrix(b.ctypes.data_as(POINTER(c_float)), 3, 2)

        c = calculate_matrix_multiple(A, B)

        np.testing.assert_array_equal([3, 3, 3, 3], c)


if __name__ == '__main__':
    unittest.main()
