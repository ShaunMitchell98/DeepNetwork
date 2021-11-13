#include "LayerPropagator.h"

void LayerPropagator::PropagateLayer(Matrix* weights, Vector* inputLayer, Vector* bias, Vector* outputLayer) {

    _logger.LogLine("Forward propagating layer.");

    _logger.LogLine("Forward propagation input: ");
    _logger.LogMessage(inputLayer->ToString());

    *outputLayer = *weights * *inputLayer;

    _logger.LogLine("Forward propagation output: ");
    _logger.LogMessage(outputLayer->ToString());

    *outputLayer += *bias;

    _logger.LogLine("Output after adding biases: ");
    _logger.LogMessage(outputLayer->ToString());

    //_logger.LogLine("Weight Matrix");
    //_logger.LogMatrix(weights);

    outputLayer->ApplyActivation();
    _logger.LogLine("Output after activation");
    _logger.LogMessage(outputLayer->ToString());
}