import unittest
from Infrastructure.matrixMultiple import *


class MyTestCase(unittest.TestCase):
    def test_GivenSquareMatrix_ReturnsResult(self):

        a = np.array([[3, 3], [3, 3]])
        b = np.array([[2, 2], [2, 2]])

        c = calculate_matrix_multiple(a, b)

        np.testing.assert_array_equal([12, 12, 12, 12], c)

    def test_GivenNonSquareMatrix_ReturnsResult(self):

        a = np.array([[1, 1, 1], [1, 1, 1]])
        b = np.array([[1, 1], [1, 1], [1, 1]])

        c = calculate_matrix_multiple(a, b)

        np.testing.assert_array_equal([3, 3, 3, 3], c)

if __name__ == '__main__':
    unittest.main()
