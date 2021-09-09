#pragma once

#include <memory>
#include <vector>
#include "PyNet.Models/Matrix.h"

class AdjustmentCalculator
{
private:
	std::vector<std::unique_ptr<Models::Matrix>> _adjustments;
	bool _newBatch;
public:
	AdjustmentCalculator();
	void AddMatrix(int rows, int cols);
	void AddAdjustment(int matrixIndex, int row, int col, double adjustment);
	double GetAdjustment(int matrixIndex, int row, int col);
	void SetNewBatch(bool newBatch);
};

