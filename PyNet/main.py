from PyNet.PyNet.PyNetwork import PyNetwork
from PyNet.PyNet import ActivationFunctionType
import tensorflow as tf
import numpy as np
import TrainSession
import matplotlib.pyplot as plt
from LogPlotter import LogPlotter

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
numbers: list[int] = [50000]
dropoutRates: list[float] = [0.9]

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        for momentum in momenta:
            for dropoutRate in dropoutRates:
                for epoch in epochs:
                    for number in numbers:

                        network = PyNetwork(True, True)
                        network.SetVariableLearning(0.04, 0.7, 1.05)
                        network.add_layer(784, ActivationFunctionType.LOGISTIC, 0.8)
                        network.add_layer(500, ActivationFunctionType.LOGISTIC, dropoutRate)
                        network.add_layer(129, ActivationFunctionType.LOGISTIC, dropoutRate)
                        network.add_layer(10, ActivationFunctionType.LOGISTIC, dropoutRate)


                       # errors = network.train(train_images[1:number], train_labels[1:number], 10, batch_size,
                       #                        learning_rate,
                        #                       epoch,
                       #                        momentum)


                      #  print('Time: ', stop - start)
                      #  success_count: int = 0
                      #  for i in range(58000, 60000):
                      ##      predicted_output: np.ndarray = network.run(train_images[i])
                      #      predicted_index = np.argmax(predicted_output)

                       #     if predicted_index == train_labels[i]:
                        #        success_count += 1

                       # with open('successes.txt', 'a') as success_file:
                       #     success_file.write(
                        #        f'There were {success_count} successes out of 2000 tests with a learning rate of'
                        #        f' {learning_rate}, batch size of {batch_size}, momentum of {momentum}, epoch of {epoch},'
                       #         f' number of {number} and dropout rate '
                       #         f'of {dropoutRate}.\n')

                    startNumber = 0
                    endNumber = 200

                    while endNumber <= number:
                        TrainSession.TrainSession(train_images, train_labels, network, startNumber,
                                                  endNumber, batch_size, learning_rate, epoch, momentum)

                        LogPlotter.PlotLog('Example Number: (.*), Loss is (.*)', 'Change_In_Loss.png',
                                           'Change In Loss',
                                           'Loss')
                        LogPlotter.PlotLog('Example Number: (.*), Test Accuracy is (\\d*)',
                                           'Change_In_Accuracy.png',
                                           'Change In Accuracy',
                                           'Accuracy (/100)')

                        for i in range(0, 3):
                            LogPlotter.PlotLog(f'Example Number: (.*), Bias for trainable layer {i} is (.*)\n',
                                               f'Change_In_Bias_Layer{i}.png',
                                               'Change In Bias',
                                               'Bias')

                        LogPlotter.PlotLog(f'Example Number: (.*), Weights for trainable layer {i} are (.*)\n',
                                           f'Change in first weight.png',
                                           'Change in Weight',
                                           'Weight'
                                           )

                        startNumber = startNumber + 200
                        endNumber = endNumber + 200

                    network.destruct()

