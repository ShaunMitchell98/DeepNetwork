#pragma once

#include "PyNet.Models/Loss.h"

namespace PyNet::Infrastructure {
	class QuadraticLoss : public Models::Loss {
	public:

		static auto factory() {
			return new QuadraticLoss();
		}

		typedef PyNet::Models::Loss base;

		double CalculateLoss(Models::Vector& expected, Models::Vector& actual) override;
		std::unique_ptr<Models::Vector> CalculateDerivative(Models::Vector& expected, Models::Vector& actual) override;
	};
}
