#pragma once

#include <vector>
#include "Matrix.h"

namespace ActivationFunctions {

	enum class ActivationFunctionType {
		Logistic
	};

	class Activation {
	protected:
	public:
		virtual void Apply(std::vector<double>& values) = 0;
		virtual void CalculateDerivative(PyNet::Models::Matrix* input, PyNet::Models::Matrix* output) = 0;
	};
}