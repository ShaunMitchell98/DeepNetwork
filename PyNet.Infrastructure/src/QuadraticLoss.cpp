#include "QuadraticLoss.h"

namespace PyNet::Infrastructure {

	double QuadraticLoss::CalculateLoss(Models::Vector& expected, Models::Vector& actual) {

		auto difference = expected - actual;
		return 0.5 * (*difference | *difference);
	}

	std::unique_ptr<Models::Vector> QuadraticLoss::CalculateDerivative(Models::Vector& expected, Models::Vector& actual) {
		return actual - expected;
	}
}
