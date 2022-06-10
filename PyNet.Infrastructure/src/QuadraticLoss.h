#pragma once

#include "PyNet.Models/Loss.h"
#include "PyNet.Models/ILogger.h"
#include "Headers.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure {
	class EXPORT QuadraticLoss : public Loss {
	private:
	shared_ptr<ILogger> _logger;
	public:

		QuadraticLoss(shared_ptr<ILogger> logger) {
			_logger = logger;
		}

		static auto factory(shared_ptr<ILogger> logger) {
			return new QuadraticLoss(logger);
		}

		double CalculateLoss(const Matrix& expected, const Matrix& actual) const override;
		unique_ptr<Matrix> CalculateDerivative(const Matrix& expected, const Matrix& actual) const override;
		~QuadraticLoss() override = default;
	};
}
