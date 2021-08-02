import ctypes


def generate_random_weights(count):

    deep_network = ctypes.windll.LoadLibrary(r"..\x64\Release\DeepNetwork.Infrastructure.dll")

    deep_network_generate_random_weights = deep_network.generate_random_weights
    deep_network_generate_random_weights.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

    values = (ctypes.c_double * count)()

    deep_network_generate_random_weights(values, count)

    return values