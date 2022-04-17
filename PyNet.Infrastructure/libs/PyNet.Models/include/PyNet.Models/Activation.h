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
		virtual void Apply(Matrix& input) = 0;
		virtual std::unique_ptr<Matrix> CalculateDerivative(const Matrix& input) = 0;
		virtual ~Activation() = default;
	};
}