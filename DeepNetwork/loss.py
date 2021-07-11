def loss(expectedOutput, actualOutput):

    loss = 0
    for i in range(0, expectedOutput.rows * expectedOutput.cols):
        loss += (expectedOutput[i] - actualOutput[i]) ** 2

    return loss
