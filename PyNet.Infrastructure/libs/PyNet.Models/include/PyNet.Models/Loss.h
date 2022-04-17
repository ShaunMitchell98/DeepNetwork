#pragma once

#include "Vector.h"

namespace PyNet::Models {

	class Loss {
	public:
		virtual	double CalculateLoss(Vector& expected, Vector& actual) = 0;
		virtual std::unique_ptr<Vector> CalculateDerivative(Vector& expected, Vector& actual) = 0;
		virtual ~Loss() = default;
	};
}
