import PyNetwork
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0

test_images = test_images / 255.0

network = PyNetwork.PyNetwork(784)
network.add_layer(128)
network.add_layer(10)

#network = PyNetwork.PyNetwork(4)
#network.add_layer(2)
first_image = train_images[0]

#error = network.train(np.array([1, 1, 1, 1]), 5)
x =list(range(0, 10000))
errors = list()

for i in range(0, 10000):
   errors.append(network.train(train_images[i].flatten(order = 'C'), train_labels[i]))
   print(f'Loop {i}')

# print(f'Loop {i}, Error {error:.16f}')
    #if i % 100 == 0:
      #  print(f'Loop {i}, Error {error:.16f}')

plt.plot(x, errors)
plt.show()
#for j in range(0, len(test_images)):
 #   network.evaluate(test_images[i], test_labels[i])


