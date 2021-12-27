#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "PyNet.Models/ILogger.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class __declspec(dllexport) LayerPropagator {
	private:
		std::shared_ptr<ILogger> _logger;
	public:

		auto static factory(std::shared_ptr<ILogger> logger) {
			return new LayerPropagator(logger);
		}

		LayerPropagator(std::shared_ptr<ILogger> logger) : _logger(logger) {}
		void PropagateLayer(Matrix& weights, Vector& inputLayer, Vector& bias, std::unique_ptr<Vector>& outputLayer);
	};
}

