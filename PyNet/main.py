from PyNet.PyNet.PyNetwork import PyNetwork
from PyNet.PyNet import ActivationFunctionType
import tensorflow as tf
import numpy as np
import timeit

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

batch_sizes = [5]
learning_rates = [0.2]
momenta = [0.7]
epochs = 1
number: int = 30000

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for momentum in momenta:

            network = PyNetwork(False, False)
            network.SetVariableLearning(0.04, 0.7, 1.05)
            network.add_layer(784, ActivationFunctionType.LOGISTIC)
            network.add_layer(1000, ActivationFunctionType.LOGISTIC)
            network.add_layer(10, ActivationFunctionType.LOGISTIC)

            start = timeit.default_timer()
            errors = network.train(train_images[1:number], train_labels[1:number], 10, batch_size, learning_rate,
                                   epochs,
                                   momentum)
            stop = timeit.default_timer()

            print('Time: ', stop - start)
            success_count: int = 0
            for i in range(58000, 60000):
                predicted_output: np.ndarray = network.run(train_images[i])
                predicted_index = np.argmax(predicted_output)

                if predicted_index == train_labels[i]:
                    success_count += 1

            with open('successes.txt', 'a') as success_file:
                success_file.write(f'There were {success_count} successes out of 2000 tests with a learning rate of'
                    f' {learning_rate} and a batch size of {batch_size}.\n')

            network.destruct()
