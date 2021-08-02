from ctypes import *
from matrix import matrix


def normalise_layer(A):

    deep_network = windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")

    library_normalise_layer = deep_network.normalise_layer
    library_normalise_layer .argtypes = [matrix]

    library_normalise_layer(A)
