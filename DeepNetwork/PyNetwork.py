from matrix import matrix
from ctypes import *
import numpy as np
from forward_propagate_layer import forward_propagate_layer
from train_layer import train_network


class PyNetwork:

    def __init__(self, count):
        self.layers = [matrix(np.random.rand(count).ctypes.data_as(POINTER(c_float)), count, 1)]
        self.weights = []
        self.weightMatrixCount = 0
        self.layerCount = 1

    def add_layer(self, count):

        cols = self.layers[-1].rows

        layer = np.random.rand(count).ctypes.data_as(POINTER(c_float))
        self.layers.append(matrix(layer, count, 1))

        initial_matrix = np.random.rand(cols * count).ctypes.data_as(POINTER(c_float))
        self.weights.append(matrix(initial_matrix, count, cols))
        self.layerCount += 1
        self.weightMatrixCount += 1

    def run(self, input_layer):
        self.layers[0].values = input_layer.ctypes.data_as(POINTER(c_float))

        for i in range(0, len(self.layers) - 1):
            self.layers[i + 1] = forward_propagate_layer(self.weights[i], self.layers[i])

    def train(self, input_layer, expected_output):
        self.run(input_layer)
        expected_output_matrix = matrix(expected_output.ctypes.data_as(POINTER(c_float)), expected_output.size, 1)
        train_network(self, expected_output_matrix)