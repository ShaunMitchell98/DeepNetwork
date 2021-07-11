#include "forward_propagate.h"
#include "../Logging/log.h"
#include "../Matrix Multiplication/matrix_multiplication.h"
#include "../Activation Functions/logistic_function.h"

void forward_propagate_layer(matrix weights, matrix inputLayer, matrix outputLayer, activation_function activationFunction) {
    log_line("Forward propagating layer.");

    matrix_multiply(weights, inputLayer, outputLayer);

    if (activationFunction == logistic) {
        apply_logistic(outputLayer);
    }
}