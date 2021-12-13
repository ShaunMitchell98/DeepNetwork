#include "AdjustmentCalculator.h"

void AdjustmentCalculator::SetBatchSize(int batchSize) {
	_batchSize = batchSize;
}

void AdjustmentCalculator::AddMatrix(int rows, int cols) {
	_weightAdjustments.push_back(std::move(_context->GetUnique<PyNet::Models::Matrix>()));
	_biasAdjustments.push_back(std::move(_context->GetUnique<PyNet::Models::Vector>()));
}

void AdjustmentCalculator::AddWeightAdjustment(int matrixIndex, std::unique_ptr<PyNet::Models::Matrix> adjustments) {
	
	if (_newBatch) {
		_weightAdjustments[matrixIndex].reset(adjustments.get());
	}
	else {
		*_weightAdjustments[matrixIndex] += *adjustments;

	}
}

void AdjustmentCalculator::AddBiasAdjustment(int matrixIndex, double adjustment) {

	if (_newBatch) {
		_biasAdjustments[matrixIndex]->SetValue(adjustment);
	}
	else {
		_biasAdjustments[matrixIndex]->AddValue(adjustment);
	}
}


std::unique_ptr<PyNet::Models::Matrix> AdjustmentCalculator::GetWeightAdjustment(int matrixIndex) {
	return *_weightAdjustments[matrixIndex].get() / _batchSize;
}

std::unique_ptr<PyNet::Models::Vector> AdjustmentCalculator::GetBiasAdjustment(int matrixIndex) {
	return *_biasAdjustments[matrixIndex].get() / _batchSize;
}

void AdjustmentCalculator::SetNewBatch(bool newBatch) {
	_newBatch = newBatch;
}