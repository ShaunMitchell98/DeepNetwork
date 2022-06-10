#include "QuadraticLoss.h"
#include <format>

namespace PyNet::Infrastructure 
{
	double QuadraticLoss::CalculateLoss(const Matrix& expected, const Matrix& actual) const 
	{
		auto difference = expected - actual;
		auto loss =  0.5 * (*difference | *difference);
		_logger->LogLine("Loss is {}", make_format_args(loss));
		return loss;
	}

	unique_ptr<Matrix> QuadraticLoss::CalculateDerivative(const Matrix& expected, const Matrix& actual) const
	{
		return actual - expected;
	}
}
