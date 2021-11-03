#include "AdjustmentCalculator.h"

AdjustmentCalculator::AdjustmentCalculator(Settings* settings) {
	_newBatch = true;
	_batchSize = 0;
	_weightAdjustments = std::vector<std::unique_ptr<PyNet::Models::Matrix>>();
	_biasAdjustments = std::vector<std::unique_ptr<PyNet::Models::Vector>>();
	_settings = settings;
}

void AdjustmentCalculator::SetBatchSize(int batchSize) {
	_batchSize = batchSize;
}

void AdjustmentCalculator::AddMatrix(int rows, int cols) {
	_weightAdjustments.push_back(std::make_unique<PyNet::Models::Matrix>(rows, cols, _settings->CudaEnabled));
	_biasAdjustments.push_back(std::make_unique<PyNet::Models::Vector>(rows, _settings->CudaEnabled));
}

void AdjustmentCalculator::AddWeightAdjustment(int matrixIndex, PyNet::Models::Matrix* adjustments) {
	
	if (_newBatch) {
		*_weightAdjustments[matrixIndex] = *adjustments;
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


PyNet::Models::Matrix* AdjustmentCalculator::GetWeightAdjustment(int matrixIndex) {
	return &(*_weightAdjustments[matrixIndex] / _batchSize);
}

PyNet::Models::Vector* AdjustmentCalculator::GetBiasAdjustment(int matrixIndex) {
	return &(*_biasAdjustments[matrixIndex] / _batchSize);
}

void AdjustmentCalculator::SetNewBatch(bool newBatch) {
	_newBatch = newBatch;
}