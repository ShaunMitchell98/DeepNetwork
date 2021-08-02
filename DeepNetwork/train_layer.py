from ctypes import *
from matrix import matrix
from network import network
import PyNetwork
import numpy as np


def train_network(pyNetwork, expectedOutput):

    deep_network = windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")

    network_train_network = deep_network.train_network
    network_train_network.argtypes = [network, matrix]
    network_train_network.restype = c_double

    array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    array[expectedOutput] = 1
    array = array.astype(np.double)

    expectedOutput = matrix(array.ctypes.data_as(POINTER(c_double)), array.size, 1)

    inputNetwork = network()
    inputNetwork.weightMatrixCount = pyNetwork.weightMatrixCount
    inputNetwork.layerCount = pyNetwork.layerCount
    inputNetwork.layers = (matrix * pyNetwork.layerCount)(*pyNetwork.layers)
    inputNetwork.weights = (matrix * pyNetwork.weightMatrixCount)(*pyNetwork.weights)

    return network_train_network(inputNetwork, expectedOutput)