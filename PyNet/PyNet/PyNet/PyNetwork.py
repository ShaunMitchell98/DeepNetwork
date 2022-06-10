import numpy as np
import ctypes
from PyNet.PyNet.Settings import Settings
from PyNet.PyNet.NumpyArrayConversion import convert_numpy_array_to_2d_double_array


class PyNetwork:

    def __init__(self, log: bool, cudaEnabled: bool):

        self.lib = ctypes.cdll.LoadLibrary(r"..\PyNet.Infrastructure\build\Release\PyNet.Infrastructure.dll")
        self.lib.PyNetwork_Initialise.argtypes = [ctypes.c_bool, ctypes.c_bool]
        self.lib.PyNetwork_Initialise.restype = ctypes.c_void_p

        self.lib.PyNetwork_AddInputLayer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

        self.lib.PyNetwork_AddDenseLayer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.PyNetwork_AddDenseLayer.restype = ctypes.c_void_p

        self.lib.PyNetwork_AddConvolutionLayer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.PyNetwork_AddConvolutionLayer.restype = ctypes.c_void_p

        self.lib.PyNetwork_AddMaxPoolingLayer.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.lib.PyNetwork_AddDropoutLayer.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int]

        self.lib.PyNetwork_AddFlattenLayer.argtypes = [ctypes.c_void_p]

        self.lib.PyNetwork_AddSoftmaxLayer.argtypes = [ctypes.c_void_p]

        self.lib.PyNetwork_Run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        self.lib.PyNetwork_Run.restype = ctypes.POINTER(ctypes.c_double)

        self.lib.PyNetwork_Train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                             ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                                             ctypes.POINTER(Settings)]

        self.lib.PyNetwork_SetVariableLearning.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                                                           ctypes.c_double]

        self.lib.PyNetwork_Save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.PyNetwork_Load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.PyNetwork_Load.restype = ctypes.c_int

        self.lib.PyNetwork_Destruct.argtypes = [ctypes.c_void_p]

        self.obj = self.lib.PyNetwork_Initialise(log, cudaEnabled)
        self.outputNumber = 0

    def add_input_layer(self, rows: int, cols: int):
        self.lib.PyNetwork_AddInputLayer(self.obj, rows, cols)
        self.outputNumber = rows

    def add_dense_layer(self, count: int, activationFunctionType: int):
        self.lib.PyNetwork_AddDenseLayer(self.obj, count, activationFunctionType)
        self.outputNumber = count

    def add_convolution_layer(self, filterSize: int, activationFunctionType: int):
        self.lib.PyNetwork_AddConvolutionLayer(self.obj, filterSize, activationFunctionType)
        self.outputNumber = filterSize

    def add_max_pooling_layer(self, filterSize: int):
        self.lib.PyNetwork_AddMaxPoolingLayer(self.obj, filterSize)
        self.outputNumber = filterSize

    def add_dropout_layer(self, rate: float, rows: int, cols: int):
        self.lib.PyNetwork_AddDropoutLayer(self.obj, rate, rows, cols)

    def add_softmax_layer(self):
        self.lib.PyNetwork_AddSoftmaxLayer(self.obj)

    def add_flatten_layer(self):
        self.lib.PyNetwork_AddFlattenLayer(self.obj)

    def run(self, input_layer: np.ndarray) -> np.ndarray:
        results = self.lib.PyNetwork_Run(self.obj, input_layer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return np.ctypeslib.as_array(results, shape=(self.outputNumber,))

    def train(self, input_layers: np.ndarray,
              expected_outputs: np.ndarray, numberOfOutputOptions: int, settings: Settings):

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

        self.lib.PyNetwork_Train(self.obj, input_arr_ptr, expected_arr_ptr, ctypes.pointer(settings))

    def SetVariableLearning(self, errorThreshold: float, lrDecrease: float, lrIncrease: float):
        self.lib.PyNetwork_SetVariableLearning(self.obj, errorThreshold, lrDecrease, lrIncrease)

    def save(self, filePath):
        self.lib.PyNetwork_Save(self.obj, ctypes.c_char_p(filePath.encode('utf-8')))

    def load(self, filePath):
        self.outputNumber = self.lib.PyNetwork_Load(self.obj, ctypes.c_char_p(filePath.encode('utf-8')))

    def destruct(self):
        self.lib.PyNetwork_Destruct(self.obj)
