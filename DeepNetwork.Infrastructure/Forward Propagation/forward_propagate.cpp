#include "forward_propagate.h"
#include "../Logging/logger.h"
#include "../Matrix Multiplication/matrix_multiplication.h"
#include "../Activation Functions/logistic_function.h"
#include <memory>

void forward_propagate_layer(matrix weights, matrix inputLayer, matrix outputLayer, activation_function activationFunction) {
    auto logger = std::make_unique<Logger>();

    logger->LogNumber(weights.rows);
    logger->LogNumber(weights.cols);
    //logger->LogNumber(weights.values);
    logger->LogLine("Weight Matrix: ");
    //logger->LogMatrix(weights);

    logger->LogLine("Forward propagating layer.");

    matrix_multiply(weights, inputLayer, outputLayer);

    logger->LogLine("Forward propagation input: ");
    logger->LogDoubleArray(inputLayer.values, inputLayer.rows);

    logger->LogLine("Forward propagation output: ");
    logger->LogDoubleArray(outputLayer.values, outputLayer.rows);

    if (activationFunction == activation_function::logistic) {
        apply_logistic(outputLayer, logger.get());
        logger->LogLine("Output after logistic");
        logger->LogMatrix(outputLayer);
    }
}