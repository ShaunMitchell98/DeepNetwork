#include "AdjustmentCalculator.h"

namespace PyNet::Infrastructure {

	void AdjustmentCalculator::CalculateWeightAdjustment(Matrix& newAdjustment, Matrix& total) {

		auto adjustmentWithMomentum = newAdjustment * (1 - _settings->Momentum);

		if (_settings->NewBatch) {

			if (_settings->Momentum > 0) {
				adjustmentWithMomentum = *adjustmentWithMomentum - *(newAdjustment * _settings->Momentum);
			}

			total = *adjustmentWithMomentum;
		}
		else {
			total += *adjustmentWithMomentum;
		}
	}

	double AdjustmentCalculator::CalculateBiasAdjustment(double newAdjustment, double total) {

		auto adjustmentWithMomentum = newAdjustment * (1 - _settings->Momentum);

		if (_settings->NewBatch) {
			auto totalAdjustment = adjustmentWithMomentum - (newAdjustment * _settings->Momentum);
			total = totalAdjustment;
		}
		else {
			total += adjustmentWithMomentum;
		}

		return total;
	}
}