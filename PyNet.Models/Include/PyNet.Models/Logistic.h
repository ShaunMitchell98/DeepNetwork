#pragma once

#include "Activation.h"
#include <vector>

namespace ActivationFunctions {

	class Logistic : public Activation {
	public:
		void Apply(std::vector<double>& values);
		double CalculateDerivative(double input);
	};

}
