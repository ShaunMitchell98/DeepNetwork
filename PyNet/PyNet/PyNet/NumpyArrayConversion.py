import ctypes
from numpy import ndarray


def convert_numpy_array_to_2d_double_array(numpy_array: ndarray) -> ctypes.POINTER(ctypes.c_double):
    arr_ptr: ctypes.POINTER(ctypes.c_double) = (ctypes.POINTER(ctypes.c_double) * numpy_array.shape[0])()

    for i, row in enumerate(numpy_array):
        arr_ptr[i] = (ctypes.c_double * numpy_array.shape[1])()

        for j, val in enumerate(row):
            arr_ptr[i][j] = val

    return arr_ptr
