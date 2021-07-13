#include "forward_propagate.h"
#include "../Logging/logger.h"
#include "../Matrix Multiplication/matrix_multiplication.h"
#include "../Activation Functions/logistic_function.h"

void forward_propagate_layer(matrix weights, matrix inputLayer, matrix outputLayer, activation_function activationFunction) {
    auto logger = new Logger();
    logger->LogLine("Forward propagating layer.");

    matrix_multiply(weights, inputLayer, outputLayer);

    if (activationFunction == activation_function::logistic) {
        apply_logistic(outputLayer);
    }

    delete logger;
}