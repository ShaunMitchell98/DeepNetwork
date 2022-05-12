#pragma once

#include "Matrix.h"

namespace PyNet::Models {

	class Loss {
	public:
		virtual	double CalculateLoss(Matrix& expected, Matrix& actual) = 0;
		virtual std::unique_ptr<Matrix> CalculateDerivative(Matrix& expected, Matrix& actual) = 0;
		virtual ~Loss() = default;
	};
}
