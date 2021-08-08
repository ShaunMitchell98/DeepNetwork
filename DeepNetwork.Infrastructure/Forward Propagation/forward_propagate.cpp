#include "forward_propagate.h"
#include "../Logging/logger.h"
#include "../Matrix Multiplication/matrix_multiplication.h"
#include "../Activation Functions/logistic_function.h"
#include <memory>

void forward_propagate_layer(Matrix* weights, Matrix* inputLayer, Matrix* outputLayer, activation_function activationFunction) {

    auto logger = std::make_unique<Logger>();
    //logger->LogNumber(weights->Rows);
    //logger->LogNumber(weights->Cols);
    //logger->LogNumber(weights.values);
    //logger->LogLine("Weight Matrix: ");
    //logger->LogMatrix(weights);

    logger->LogLine("Forward propagating layer.");

    matrix_multiply(weights, inputLayer, outputLayer);

    logger->LogLine("Forward propagation input: ");
    logger->LogDoubleArray(inputLayer->Values.data(), inputLayer->Rows);

    logger->LogLine("Forward propagation output: ");
    logger->LogDoubleArray(outputLayer->Values.data(), outputLayer->Rows);

    if (activationFunction == activation_function::logistic) {
        apply_logistic(outputLayer, logger.get());
        logger->LogLine("Output after logistic");
        logger->LogMatrix(outputLayer);
    }
}