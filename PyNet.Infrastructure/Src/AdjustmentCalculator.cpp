#include "AdjustmentCalculator.h"

namespace PyNet::Infrastructure {

	void AdjustmentCalculator::CalculateWeightAdjustment(Matrix& newAdjustment, Matrix& total) {

		//auto adjustmentWithMomentum = newAdjustment * (1 - _settings->Momentum);

		if (_trainingState->NewBatch) {

	/*		if (_settings->Momentum > 0) {
				adjustmentWithMomentum = *adjustmentWithMomentum - *(total * _settings->Momentum);
			}*/

			//total = *adjustmentWithMomentum;
			total = newAdjustment;
		}
		else {
			//total += *adjustmentWithMomentum;
			total += newAdjustment;
		}
	}

	double AdjustmentCalculator::CalculateBiasAdjustment(double newAdjustment, double total) {

		auto adjustmentWithMomentum = newAdjustment * (1 - _settings->Momentum);

		if (_trainingState->NewBatch) {
			auto totalAdjustment = adjustmentWithMomentum - (total * _settings->Momentum);
			total = totalAdjustment;
			//total = newAdjustment;
		}
		else {
			total += adjustmentWithMomentum;
			//total += newAdjustment;
		}

		return total;
	}
}