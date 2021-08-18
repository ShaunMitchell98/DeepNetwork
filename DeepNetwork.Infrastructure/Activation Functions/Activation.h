#pragma once

#include <vector>

namespace ActivationFunctions {

	enum class ActivationFunctionType {
		Logistic
	};

	class Activation {
	protected:
	public:
		virtual void Apply(std::vector<double> values) = 0;
		virtual double CalculateDerivative(double input) = 0;
	};
}