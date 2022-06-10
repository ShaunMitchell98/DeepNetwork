import ctypes as ct


class Settings(ct.Structure):
    _fields_ = [("LoggingEnabled", ct.c_bool),
                ("CudaEnabled", ct.c_bool),
                ("RunMode", ct.c_int),
                ("BaseLearningRate", ct.c_double),
                ("BatchSize", ct.c_int),
                ("Epochs", ct.c_int),
                ("NumberOfExamples", ct.c_int),
                ("StartExampleNumber", ct.c_int),
                ("Momentum", ct.c_double)]
