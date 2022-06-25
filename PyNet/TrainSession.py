from PyNet.PyNet.PyNetwork import PyNetwork
from PyNet.PyNet.Settings import Settings
import numpy as np


def TrainSession(train_images, train_labels, network: PyNetwork, start: int, end: int, settings: Settings):

    settings.StartExampleNumber = start
    network.train(train_images[start:end], train_labels[start:end], 10, settings)

    success_count: int = 0
    for i in range(58000, 58100):
        predicted_output: np.ndarray = network.run(train_images[i])
        predicted_index = np.argmax(predicted_output)

        if predicted_index == train_labels[i]:
            success_count += 1

    with open('PyNet_Logs.txt', 'a') as success_file:
        success_file.write(
            f'Epoch: {settings.Epochs}, Example Number: {end}, Test Accuracy is {success_count}\r\n')



