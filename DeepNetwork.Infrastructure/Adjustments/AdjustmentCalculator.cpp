#include "AdjustmentCalculator.h"

AdjustmentCalculator::AdjustmentCalculator() {
	_batchNumber = 1;

	_adjustments = std::vector<std::unique_ptr<Models::Matrix>>();
}

void AdjustmentCalculator::AddMatrix(int rows, int cols) {
	_adjustments.push_back(std::make_unique<Models::Matrix>(rows, cols));
}

void AdjustmentCalculator::AddAdjustment(int matrixIndex, int row, int col, double adjustment) {

	double* currentAdjustment = _adjustments[matrixIndex]->GetAddress(row, col);
	
	if (_batchNumber == 1) {
		*currentAdjustment = adjustment;
	}
	else {
		*currentAdjustment += adjustment;
	}
}

double AdjustmentCalculator::GetAdjustment(int matrixIndex, int row, int col) {
	return _adjustments[matrixIndex]->GetValue(row, col);
}

void AdjustmentCalculator::IncrementBatchNumber() {
	_batchNumber++;
}