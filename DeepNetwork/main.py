from PyNetwork import PyNetwork
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images / 255.0

test_images = test_images / 255.0

network = PyNetwork(784)
network.add_layer(128)
network.add_layer(10)

batch_size = 5

first_image = train_images[0]

errors = network.train(train_images, train_labels, 10, batch_size)

x = list(range(0, train_images.shape[0]))
plt.plot(x, errors)
plt.show()
