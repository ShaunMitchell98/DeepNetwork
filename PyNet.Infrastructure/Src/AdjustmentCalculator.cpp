#include "AdjustmentCalculator.h"

namespace PyNet::Infrastructure {

	void AdjustmentCalculator::SetBatchSize(int batchSize) {
		_batchSize = batchSize;
	}

	void AdjustmentCalculator::SetMomentum(int momentum) {
		_momentum = momentum;
	}

	void AdjustmentCalculator::AddMatrix(int rows, int cols) {
		auto weightMatrix = _context->GetUnique<Matrix>();
		weightMatrix->Initialise(rows, cols, false);
		_weightAdjustments.push_back(move(weightMatrix));

		auto biasVector = _context->GetUnique<Vector>();
		biasVector->Initialise(rows, false);
		_biasAdjustments.push_back(move(biasVector));
	}

	void AdjustmentCalculator::AddWeightAdjustment(int matrixIndex, unique_ptr<Matrix> adjustments) {

		auto adjustmentWithMomentum = *adjustments * (1 - _momentum);

		if (_newBatch) {

			if (_momentum > 0) {
				adjustmentWithMomentum = *adjustmentWithMomentum - *(*_weightAdjustments[matrixIndex] * _momentum);
			}

			_weightAdjustments[matrixIndex] = move(adjustmentWithMomentum);
		}
		else {
			*_weightAdjustments[matrixIndex] += *adjustmentWithMomentum;
		}
	}

	void AdjustmentCalculator::AddBiasAdjustment(int matrixIndex, double adjustment) {

		auto adjustmentWithMomentum = adjustment * (1 - _momentum);

		if (_newBatch) {
			auto totalAdjustment = adjustmentWithMomentum - ((*_biasAdjustments[matrixIndex])[0] * _momentum);
			_biasAdjustments[matrixIndex]->SetValue(totalAdjustment);
		}
		else {
			_biasAdjustments[matrixIndex]->AddValue(adjustmentWithMomentum);
		}
	}

	unique_ptr<Matrix> AdjustmentCalculator::GetWeightAdjustment(int matrixIndex) const {
		return move(*_weightAdjustments[matrixIndex] / _batchSize);
	}

	unique_ptr<Vector> AdjustmentCalculator::GetBiasAdjustment(int matrixIndex) const {
		return move(*_biasAdjustments[matrixIndex] / _batchSize);
	}

	void AdjustmentCalculator::SetNewBatch(bool newBatch) {
		_newBatch = newBatch;
	}
}