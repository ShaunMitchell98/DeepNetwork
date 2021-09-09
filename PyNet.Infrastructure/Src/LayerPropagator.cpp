#include "LayerPropagator.h"
#include "matrix_multiplication.h"
#include "PyNet.Models/Logistic.h"
#include <memory>

using namespace ActivationFunctions;

LayerPropagator::LayerPropagator(std::shared_ptr<ILogger> logger) {
    _logger = logger;
}

void LayerPropagator::PropagateLayer(Matrix* weights, Models::Vector* inputLayer, Models::Vector* outputLayer) {

    _logger->LogLine("Forward propagating layer.");

    matrix_multiply(weights, inputLayer, outputLayer);

    _logger->LogLine("Weight Matrix");
    //_logger->LogMatrix(weights);

    _logger->LogLine("Forward propagation input: ");
    _logger->LogVector(inputLayer->Values);

    _logger->LogLine("Forward propagation output: ");
    _logger->LogVector(outputLayer->Values);

    outputLayer->ApplyActivation();
    _logger->LogLine("Output after activation");
    _logger->LogMatrix(outputLayer);
}