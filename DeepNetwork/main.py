from network import network
import numpy as np

myNetwork = network(5)
myNetwork.add_layer(4)
myNetwork.add_layer(6)

result = myNetwork.run()

values = np.ctypeslib.as_array(result.values, shape=(6,))

e = 5


