#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "PyNet.Models/ILogger.h"

using namespace PyNet::Models;

class __declspec(dllexport) LayerPropagator {
private:
	ILogger& _logger;
public:

	auto static factory(ILogger& logger) {
		return new LayerPropagator(logger);
	}

	LayerPropagator(ILogger& logger) : _logger(logger) {}
	void PropagateLayer(Matrix& weights, Vector& inputLayer, Vector& bias, Vector& outputLayer);
};