from matrix import matrix
import numpy as np
from matrixMultiple import calculate_matrix_multiple
from ctypes import *

class network:

    def __init__(self, count):
        self.layers = [matrix(np.random.rand(count).ctypes.data_as(POINTER(c_float)), count, 1)]
        self.matrices = []

    def add_layer(self, count):
        last_element = self.layers[-1]

        layer = np.random.rand(count).ctypes.data_as(POINTER(c_float))
        self.layers.append(matrix(layer, count, 1))

        initial_matrix = np.random.rand(last_element.rows * count).ctypes.data_as(POINTER(c_float))
        self.matrices.append(matrix(initial_matrix, count, last_element.rows))

    def run(self):
        for i in range(0, len(self.layers)-1):
            layer = self.layers[i]
            matrix = self.matrices[i]
            self.layers[i+1] = calculate_matrix_multiple(matrix, layer)
        return self.layers[-1]