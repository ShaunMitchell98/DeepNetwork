from matrix import matrix
from ctypes import *


class network(Structure):
    _fields_ = [("layers", POINTER(matrix)),
                ("weights", POINTER(matrix)),
                ("layerCount", c_int),
                ("weightMatrixCount", c_int)]




