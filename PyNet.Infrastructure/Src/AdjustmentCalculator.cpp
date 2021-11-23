#include "AdjustmentCalculator.h"

void AdjustmentCalculator::SetBatchSize(int batchSize) {
	_batchSize = batchSize;
}

void AdjustmentCalculator::AddMatrix(int rows, int cols) {
	_weightAdjustments.push_back(_context.get<PyNet::Models::Matrix>());
	_biasAdjustments.push_back(_context.get<PyNet::Models::Vector>());
}

void AdjustmentCalculator::AddWeightAdjustment(int matrixIndex, PyNet::Models::Matrix& adjustments) {
	
	if (_newBatch) {
		_weightAdjustments[matrixIndex] = adjustments;
	}
	else {
		_weightAdjustments[matrixIndex].get() += adjustments;
	}
}

void AdjustmentCalculator::AddBiasAdjustment(int matrixIndex, double adjustment) {

	if (_newBatch) {
		_biasAdjustments[matrixIndex].get().SetValue(adjustment);
	}
	else {
		_biasAdjustments[matrixIndex].get().AddValue(adjustment);
	}
}


PyNet::Models::Matrix& AdjustmentCalculator::GetWeightAdjustment(int matrixIndex) {
	return _weightAdjustments[matrixIndex].get() / _batchSize;
}

PyNet::Models::Vector& AdjustmentCalculator::GetBiasAdjustment(int matrixIndex) {
	return _biasAdjustments[matrixIndex].get() / _batchSize;
}

void AdjustmentCalculator::SetNewBatch(bool newBatch) {
	_newBatch = newBatch;
}