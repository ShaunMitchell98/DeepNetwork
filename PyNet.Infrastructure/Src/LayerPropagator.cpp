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

    _logger->LogLine("Forward propagation input: ");
    _logger->LogDoubleArray(inputLayer->GetAddress(0), inputLayer->Rows);

    _logger->LogLine("Forward propagation output: ");
    _logger->LogDoubleArray(outputLayer->GetAddress(0), outputLayer->Rows);

    outputLayer->ApplyActivation();
    _logger->LogLine("Output after activation");
    _logger->LogMatrix(outputLayer);
}