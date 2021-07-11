import PyNetwork
import numpy as np


network = PyNetwork.PyNetwork(5)
network.add_layer(4)
network.add_layer(6)

# result = myNetwork.run(np.array([1, 2, 3, 4, 5]))

network.train(np.array([1, 2, 3, 4, 5]), np.array([8, 7, 6, 5, 4, 3]))
e = 5


