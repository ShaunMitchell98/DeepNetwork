#pragma once
#include "PyNetwork.h"
#include "Headers.h"
#include "PyNet.Models/ILogger.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class EXPORT BackPropagator {

		private:
		shared_ptr<ILogger> _logger;
	public:

		static auto factory(shared_ptr<ILogger> logger) {
			return new BackPropagator(logger);
		}

		BackPropagator(shared_ptr<ILogger> logger) : _logger(logger) {}

		void Propagate(const PyNetwork& pyNetwork, Matrix& lossDerivative) const;
	};
}
