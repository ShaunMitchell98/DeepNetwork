#pragma once

#include "PyNet.Models/Loss.h"
#include "Headers.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure {
	class EXPORT QuadraticLoss : public Loss {
	public:

		static auto factory() {
			return new QuadraticLoss();
		}

		double CalculateLoss(const Matrix& expected, const Matrix& actual) const override;
		unique_ptr<Matrix> CalculateDerivative(const Matrix& expected, const Matrix& actual) const override;
		~QuadraticLoss() override = default;
	};
}
