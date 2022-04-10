#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "PyNet.Models/ILogger.h"

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	class LayerPropagator {
	private:
		shared_ptr<ILogger> _logger;
	public:

		auto static factory(shared_ptr<ILogger> logger) {
			return new LayerPropagator(logger);
		}

		LayerPropagator(shared_ptr<ILogger> logger) : _logger(logger) {}
		void PropagateLayer(const Matrix& weights, const Vector& inputLayer, const Vector& bias, Vector& outputLayer) const;
	};
}

