#include "AdjustmentCalculator.h"

AdjustmentCalculator::AdjustmentCalculator(int batchSize, int layerCount) {
	_batchNumber = 1;
	_batchSize = batchSize;

	_adjustments = std::vector<std::unique_ptr<Models::Matrix>>(layerCount - 1);
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