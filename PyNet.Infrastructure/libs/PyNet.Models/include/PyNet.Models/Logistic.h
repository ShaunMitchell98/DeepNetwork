#pragma once

#include "Activation.h"

namespace PyNet::Models {

	class Logistic : public Activation {

		virtual void Apply(Matrix& input) = 0;
		virtual unique_ptr<Matrix> CalculateDerivative(const Matrix& input) = 0;
	};
}
