#include "QuadraticLoss.h"

namespace PyNet::Infrastructure {

	double QuadraticLoss::CalculateLoss(Matrix& expected, Matrix& actual) {

		auto difference = expected - actual;
		return 0.5 * (*difference | *difference);
	}

	unique_ptr<Matrix> QuadraticLoss::CalculateDerivative(Matrix& expected, Matrix& actual) {
		return actual - expected;
	}
}
