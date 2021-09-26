#include "LayerPropagator.h"
#include "matrix_multiplication.h"
#include "PyNet.Models/Logistic.h"
#include <memory>

using namespace ActivationFunctions;

LayerPropagator::LayerPropagator(std::shared_ptr<ILogger> logger) {
    _logger = logger;
}

void LayerPropagator::PropagateLayer(Matrix* weights, Models::Vector* inputLayer, Models::Vector* biases, Models::Vector* outputLayer) {

    _logger->LogLine("Forward propagating layer.");

    _logger->LogLine("Forward propagation input: ");
    _logger->LogVector(inputLayer->Values);

    matrix_multiply(weights, inputLayer, outputLayer);

    _logger->LogLine("Forward propagation output: ");
    _logger->LogVector(outputLayer->Values);

    vector_add(outputLayer, biases, outputLayer);

    _logger->LogLine("Output after adding biases: ");
    _logger->LogVector(outputLayer->Values);

    _logger->LogLine("Weight Matrix");
    //_logger->LogMatrix(weights);

    outputLayer->ApplyActivation();
    _logger->LogLine("Output after activation");
    _logger->LogMatrix(outputLayer);
}