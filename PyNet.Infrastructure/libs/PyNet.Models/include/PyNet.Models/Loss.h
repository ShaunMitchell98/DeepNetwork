#pragma once

#include "Matrix.h"

namespace PyNet::Models {

	class Loss {
	public:
		virtual	double CalculateLoss(const Matrix& expected, const Matrix& actual) const = 0;
		virtual unique_ptr<Matrix> CalculateDerivative(const Matrix& expected, const Matrix& actual) const = 0;
		virtual ~Loss() = default;
	};
}
