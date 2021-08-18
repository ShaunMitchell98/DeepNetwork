#pragma once

#include "../Models/Matrix.h"
#include "../Activation Functions/Activation.h"
#include "../Models/Vector.h"
#include "../Logging/ILogger.h"

class LayerPropagator {
private:
	std::shared_ptr<ILogger> _logger;
public:
	LayerPropagator(std::shared_ptr<ILogger> logger);
	void PropagateLayer(Matrix* weights, Models::Vector* inputLayer, Models::Vector* outputLayer);
};