#pragma once

#include "PyNet.Models/Activation.h"

namespace ActivationFunctions {

	class CpuLogistic : public Activation {
	public:
		void Apply(PyNet::Models::Matrix* input);
		void CalculateDerivative(PyNet::Models::Matrix* input, PyNet::Models::Matrix* output);
	};
}
