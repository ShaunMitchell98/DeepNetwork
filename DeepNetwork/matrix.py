from ctypes import *

class matrix(Structure):
    _fields_ = [("values", POINTER(c_float)),
                ("rows", c_int),
                ("cols", c_int)]

