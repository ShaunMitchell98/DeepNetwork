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

		double CalculateLoss(Matrix& expected, Matrix& actual) override;
		unique_ptr<Matrix> CalculateDerivative(Matrix& expected, Matrix& actual) override;
		~QuadraticLoss() override = default;
	};
}
