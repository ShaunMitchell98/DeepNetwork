#include "AdjustmentCalculator.h"

void AdjustmentCalculator::SetBatchSize(int batchSize) {
	_batchSize = batchSize;
}

void AdjustmentCalculator::AddMatrix(int rows, int cols) {
	auto weightMatrix = _context->GetUnique<PyNet::Models::Matrix>();
	weightMatrix->Initialise(rows, cols, false);
	_weightAdjustments.push_back(std::move(weightMatrix));

	auto biasVector = _context->GetUnique<PyNet::Models::Vector>();
	biasVector->Initialise(rows, false);
	_biasAdjustments.push_back(std::move(biasVector));
}

void AdjustmentCalculator::AddWeightAdjustment(int matrixIndex, std::unique_ptr<PyNet::Models::Matrix> adjustments) {
	
	if (_newBatch) {
		_weightAdjustments[matrixIndex] = std::move(adjustments);
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
	return std::move(*_weightAdjustments[matrixIndex] / _batchSize);
}

std::unique_ptr<PyNet::Models::Vector> AdjustmentCalculator::GetBiasAdjustment(int matrixIndex) {
	return std::move(*_biasAdjustments[matrixIndex] / _batchSize);
}

void AdjustmentCalculator::SetNewBatch(bool newBatch) {
	_newBatch = newBatch;
}