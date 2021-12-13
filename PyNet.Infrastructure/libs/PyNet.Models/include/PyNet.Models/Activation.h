#pragma once

#include "Matrix.h"
#include <memory>

namespace PyNet::Models {

	enum class ActivationFunctionType {
		Logistic
	};

	class Activation {
	protected:
	public:
		virtual void Apply(PyNet::Models::Matrix& input) = 0;
		virtual std::unique_ptr<PyNet::Models::Matrix> CalculateDerivative(PyNet::Models::Matrix& input) = 0;
	};
}