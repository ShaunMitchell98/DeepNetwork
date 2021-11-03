#include "LayerPropagator.h"

LayerPropagator::LayerPropagator(ILogger* logger) {
    _logger = logger;
}

void LayerPropagator::PropagateLayer(PyNet::Models::Matrix* weights, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* bias, PyNet::Models::Vector* outputLayer) {

    _logger->LogLine("Forward propagating layer.");

    _logger->LogLine("Forward propagation input: ");
    _logger->LogVector(inputLayer->Values);

    *outputLayer = *weights * *inputLayer;

    _logger->LogLine("Forward propagation output: ");
    _logger->LogVector(outputLayer->Values);

    *outputLayer += *bias;

    _logger->LogLine("Output after adding biases: ");
    _logger->LogVector(outputLayer->Values);

    //_logger->LogLine("Weight Matrix");
    //_logger->LogMatrix(weights);

    outputLayer->ApplyActivation();
    _logger->LogLine("Output after activation");
    _logger->LogMatrix(outputLayer);
}