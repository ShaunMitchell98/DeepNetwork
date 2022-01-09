#include "LayerPropagator.h"

namespace PyNet::Infrastructure {

    void LayerPropagator::PropagateLayer(const Matrix& weights, const Vector& inputLayer, const Vector& bias, Vector& outputLayer) const {

        _logger->LogLine("Forward propagating layer.");

        _logger->LogLine("Forward propagation input: ");
        _logger->LogMessage(inputLayer.ToString());

        outputLayer = *(weights * inputLayer);

        _logger->LogLine("Forward propagation output: ");
        _logger->LogMessage(outputLayer.ToString());

        outputLayer += bias;

        _logger->LogLine("Output after adding biases: ");
        _logger->LogMessage(outputLayer.ToString());

        outputLayer.ApplyActivation();
        _logger->LogLine("Output after activation");
        _logger->LogMessage(outputLayer.ToString());
    }
}
