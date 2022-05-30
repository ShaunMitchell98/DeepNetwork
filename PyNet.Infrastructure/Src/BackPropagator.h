#pragma once
#include "PyNetwork.h"
#include "Headers.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class EXPORT BackPropagator {
	public:

		static auto factory() {
			return new BackPropagator();
		}

		void Propagate(const PyNetwork& pyNetwork, Matrix& lossDerivative) const;
	};
}
