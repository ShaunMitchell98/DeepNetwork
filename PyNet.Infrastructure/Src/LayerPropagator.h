#pragma once

#include "PyNet.Models/Matrix.h"
#include "Activation.h"
#include "PyNet.Models/Vector.h"
#include "ILogger.h"

class LayerPropagator {
private:
	std::shared_ptr<ILogger> _logger;
public:
	LayerPropagator(std::shared_ptr<ILogger> logger);
	void PropagateLayer(Matrix* weights, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* biases, PyNet::Models::Vector* outputLayer);
};