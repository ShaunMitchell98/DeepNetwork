#pragma once

#include "Activation.h"

namespace ActivationFunctions {

	class Logistic : public Activation {
	public:
		void Apply(std::vector<double>& values);
		void CalculateDerivative(PyNet::Models::Matrix* input, PyNet::Models::Matrix* output);
	};

}
