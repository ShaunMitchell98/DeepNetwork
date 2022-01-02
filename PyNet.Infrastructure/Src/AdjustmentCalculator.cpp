#include "AdjustmentCalculator.h"

void AdjustmentCalculator::SetBatchSize(int batchSize) {
	_batchSize = batchSize;
}

void AdjustmentCalculator::SetMomentum(int momentum) {
	_momentum = momentum;
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
	
	auto adjustmentWithMomentum = *adjustments * (1 - _momentum);

	if (_newBatch) {
		*_weightAdjustments[matrixIndex] = *(*adjustmentWithMomentum - *(*_weightAdjustments[matrixIndex] * _momentum));
	}
	else {
		*_weightAdjustments[matrixIndex] += *adjustmentWithMomentum;
	}
}

void AdjustmentCalculator::AddBiasAdjustment(int matrixIndex, double adjustment) {

	auto adjustmentWithMomentum = adjustment * (1 - _momentum);

	if (_newBatch) {
		auto totalAdjustment = adjustmentWithMomentum - (_biasAdjustments[matrixIndex]->GetValue(0) * _momentum);
		_biasAdjustments[matrixIndex]->SetValue(totalAdjustment);
	}
	else {
		_biasAdjustments[matrixIndex]->AddValue(adjustmentWithMomentum);
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