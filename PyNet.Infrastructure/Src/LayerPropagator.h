#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "PyNet.Models/ILogger.h"

class LayerPropagator {
private:
	ILogger* _logger;
public:

	auto static factory(ILogger* logger) {
		return new LayerPropagator { logger };
	}

	LayerPropagator(ILogger* logger);

	void PropagateLayer(PyNet::Models::Matrix* weights, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* bias, PyNet::Models::Vector* outputLayer);
};