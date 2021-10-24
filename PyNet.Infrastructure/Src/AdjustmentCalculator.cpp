#include "AdjustmentCalculator.h"

AdjustmentCalculator::AdjustmentCalculator(bool cudaEnabled) {
	_newBatch = true;
	_batchSize = 0;
	_weightAdjustments = std::vector<std::unique_ptr<PyNet::Models::Matrix>>();
	_biasAdjustments = std::vector<std::unique_ptr<PyNet::Models::Vector>>();
	_cudaEnabled = cudaEnabled;
}

void AdjustmentCalculator::SetBatchSize(int batchSize) {
	_batchSize = batchSize;
}

void AdjustmentCalculator::AddMatrix(int rows, int cols) {
	_weightAdjustments.push_back(std::make_unique<PyNet::Models::Matrix>(rows, cols, _cudaEnabled));
	_biasAdjustments.push_back(std::make_unique<PyNet::Models::Vector>(rows, _cudaEnabled));
}

void AdjustmentCalculator::AddWeightAdjustment(int matrixIndex, int row, int col, double adjustment) {
	
	if (_newBatch) {
		_weightAdjustments[matrixIndex]->SetValue(row, col, adjustment);
	}
	else {
		_weightAdjustments[matrixIndex]->SetValue(row, col, _weightAdjustments[matrixIndex]->GetValue(row, col) + adjustment);
	}
}
void AdjustmentCalculator::AddBiasAdjustment(int matrixIndex, int row, double adjustment) {

	if (_newBatch) {
		_biasAdjustments[matrixIndex]->SetValue(row, adjustment);
	}
	else {
		_biasAdjustments[matrixIndex]->SetValue(row, _biasAdjustments[matrixIndex]->GetValue(row) + adjustment);
	}
}


double AdjustmentCalculator::GetWeightAdjustment(int matrixIndex, int row, int col) {
	return _weightAdjustments[matrixIndex]->GetValue(row, col) / _batchSize;
}

double AdjustmentCalculator::GetBiasAdjustment(int matrixIndex, int row) {
	return _biasAdjustments[matrixIndex]->GetValue(row) / _batchSize;
}

void AdjustmentCalculator::SetNewBatch(bool newBatch) {
	_newBatch = newBatch;
}