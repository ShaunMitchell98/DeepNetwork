from PyNet.PyNet.PyNetwork import PyNetwork
from PyNet.PyNet import ActivationFunctionType
import tensorflow as tf
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0

test_images = test_images / 255.0

network = PyNetwork(784, False)
network.add_layer(500, ActivationFunctionType.LOGISTIC)
network.add_layer(129, ActivationFunctionType.LOGISTIC)
network.add_layer(10,  ActivationFunctionType.LOGISTIC)

batch_size: int = 20
learning_rate: float = 0.1
number: int = 60000

errors = network.train(train_images[1:number], train_labels[1:number], 10, batch_size, learning_rate)
x = list(range(0, train_images[1:number].shape[0]))
plt.title("Change in Error")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.plot(x, errors)
plt.show()
