#pragma once

#include "Matrix.h"

namespace PyNet::Models {

	enum class ActivationFunctionType {
		Logistic
	};

	class Activation {
	protected:
	public:
		virtual void Apply(PyNet::Models::Matrix& input) = 0;
		virtual void CalculateDerivative(PyNet::Models::Matrix& input, PyNet::Models::Matrix& output) = 0;
	};
}