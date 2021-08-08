from NumpyArrayConversion import *

lib = ctypes.windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")


class PyNetwork:

    def __init__(self, count: int):

        lib.PyNetwork_New.argtypes = [ctypes.c_int]
        lib.PyNetwork_New.restype = ctypes.c_void_p

        lib.PyNetwork_AddLayer.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.PyNetwork_AddLayer.restype = ctypes.c_void_p

        lib.PyNetwork_Run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        lib.PyNetwork_Run.restype = ctypes.c_void_p

        lib.PyNetwork_Train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int, ctypes.c_int]

        lib.PyNetwork_Train.restype = ctypes.POINTER(ctypes.c_double)

        self.obj = lib.PyNetwork_New(count)

    def add_layer(self, count: int):
        lib.PyNetwork_AddLayer(self.obj, count)

    def run(self, input_layer: np.ndarray):
        lib.PyNetwork_Run(self.obj, input_layer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def train(self, input_layers: np.ndarray,
              expected_outputs: np.ndarray, numberOfOutputOptions: int, batch_size: int):

        flattened_array = np.zeros(shape=(input_layers.shape[0], input_layers.shape[1] * input_layers.shape[2]))
        for j in range(0, input_layers.shape[0]):
            flattened_array[j] = input_layers[j].flatten(order='C')

        input_arr_ptr = convert_numpy_array_to_2d_double_array(flattened_array)

        expected_arrays = np.zeros(shape=(expected_outputs.shape[0], numberOfOutputOptions))
        for i in range(0, expected_outputs.shape[0]):
            expected_array = np.zeros(numberOfOutputOptions, dtype=np.double, order='C')
            expected_array[expected_outputs[i]] = 1
            expected_arrays[i] = expected_array

        expected_arr_ptr = convert_numpy_array_to_2d_double_array(expected_arrays)

        errors=  lib.PyNetwork_Train(self.obj, input_arr_ptr, expected_arr_ptr, input_layers.shape[0], batch_size)
        return np.ctypeslib.as_array(errors, shape=(input_layers.shape[0],))
