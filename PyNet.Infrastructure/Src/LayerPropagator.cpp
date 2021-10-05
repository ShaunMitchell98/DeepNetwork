#include "LayerPropagator.h"
#include "PyNet.Models/Logistic.h"
#include <memory>

using namespace ActivationFunctions;

LayerPropagator::LayerPropagator(std::shared_ptr<ILogger> logger) {
    _logger = logger;
}

void LayerPropagator::PropagateLayer(Matrix* weights, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* biases, PyNet::Models::Vector* outputLayer) {

    _logger->LogLine("Forward propagating layer.");

    _logger->LogLine("Forward propagation input: ");
    _logger->LogVector(inputLayer->Values);

    *outputLayer = *(Vector*)(&(*weights * *inputLayer));

    _logger->LogLine("Forward propagation output: ");
    _logger->LogVector(outputLayer->Values);

    *outputLayer += *biases;

    _logger->LogLine("Output after adding biases: ");
    _logger->LogVector(outputLayer->Values);

    _logger->LogLine("Weight Matrix");
    //_logger->LogMatrix(weights);

    outputLayer->ApplyActivation();
    _logger->LogLine("Output after activation");
    _logger->LogMatrix(outputLayer);
}