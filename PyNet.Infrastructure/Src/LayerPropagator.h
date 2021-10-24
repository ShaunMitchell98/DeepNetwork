#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "PyNet.Models/ILogger.h"
#include <memory>

class LayerPropagator {
private:
	std::shared_ptr<ILogger> _logger;
public:
	LayerPropagator(std::shared_ptr<ILogger> logger);
	void PropagateLayer(PyNet::Models::Matrix* weights, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* biases, PyNet::Models::Vector* outputLayer);
};