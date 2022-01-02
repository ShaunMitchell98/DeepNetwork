import ctypes


class VariableLearningSettings(ctypes.Structure):
    _fields_ = [
        ("ErrorThreshold", ctypes.c_double),
        ("LRDecrease", ctypes.c_double),
        ("LRIncrease", ctypes.c_double)
    ]
