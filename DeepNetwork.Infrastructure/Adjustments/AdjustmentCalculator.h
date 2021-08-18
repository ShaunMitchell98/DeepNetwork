#pragma once

#include <memory>
#include <vector>
#include "../Models/Matrix.h"

class AdjustmentCalculator
{
private:
	std::vector<std::unique_ptr<Models::Matrix>> _adjustments;
	int _batchNumber;
	int _batchSize;
public:
	AdjustmentCalculator(int batchSize, int layerCount);
	void AddAdjustment(int matrixIndex, int row, int col, double adjustment);
	double GetAdjustment(int matrixIndex, int row, int col);
};

