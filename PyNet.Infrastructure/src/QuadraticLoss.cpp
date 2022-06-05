#include "QuadraticLoss.h"

namespace PyNet::Infrastructure 
{
	double QuadraticLoss::CalculateLoss(const Matrix& expected, const Matrix& actual) const 
	{
		auto difference = expected - actual;
		return 0.5 * (*difference | *difference);
	}

	unique_ptr<Matrix> QuadraticLoss::CalculateDerivative(const Matrix& expected, const Matrix& actual) const
	{
		return actual - expected;
	}
}
