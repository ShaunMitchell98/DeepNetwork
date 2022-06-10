from PyNet.PyNet.PyNetwork import PyNetwork
import numpy as np


def TrainSession(train_images, train_labels, network: PyNetwork, start: int, end: int, batch_size, learning_rate, epoch, momentum):

    network.train(train_images[start:end], train_labels[start:end], 10, batch_size, learning_rate, epoch, momentum, start)

    success_count: int = 0
    for i in range(58000, 58100):
        predicted_output: np.ndarray = network.run(train_images[i])
        predicted_index = np.argmax(predicted_output)

        if predicted_index == train_labels[i]:
            success_count += 1

    with open('PyNet_Logs.txt', 'a') as success_file:
        success_file.write(
            f'Epoch: {epoch}, Example Number: {end}, Test Accuracy is {success_count}\r\n')



