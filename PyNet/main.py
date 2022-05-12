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

batch_sizes: list[int] = [1]
learning_rates: list[float] = [0.05]
momenta: list[float] = [0.7]
epochs: list[int] = [1]
numbers: list[int] = [20000]
dropoutRates: list[float] = [0.9]

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for momentum in momenta:
            for dropoutRate in dropoutRates:
                for epoch in epochs:
                    for number in numbers:

                        network = PyNetwork(False, True)
                        network.SetVariableLearning(0.04, 0.7, 1.05)
                        network.add_input_layer(784, 1)
                        network.add_dropout_layer(0.8, 784, 1)
                        network.add_dense_layer(500, ActivationFunctionType.LOGISTIC)
                        network.add_dropout_layer(dropoutRate, 500, 1)
                        network.add_dense_layer(129, ActivationFunctionType.LOGISTIC)
                        network.add_dropout_layer(dropoutRate, 129, 1)
                        network.add_dense_layer(10, ActivationFunctionType.LOGISTIC)

                        start = timeit.default_timer()
                        network.train(train_images[1:number], train_labels[1:number], 10, batch_size,
                                               learning_rate,
                                               epoch,
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
                            success_file.write(
                                f'There were {success_count} successes out of 2000 tests with a learning rate of'
                                f' {learning_rate}, batch size of {batch_size}, momentum of {momentum}, epoch of {epoch},'
                                f' number of {number} and dropout rate '
                                f'of {dropoutRate}.\n')

                        network.destruct()

