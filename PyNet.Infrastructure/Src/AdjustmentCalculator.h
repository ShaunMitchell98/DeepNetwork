#pragma once

#include <memory>
#include <vector>
#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "Settings.h"

class AdjustmentCalculator
{
private:
	std::vector<std::unique_ptr<PyNet::Models::Matrix>> _weightAdjustments;
	std::vector<std::unique_ptr<PyNet::Models::Vector>> _biasAdjustments;
	bool _newBatch;
	int _batchSize;
	Settings* _settings;
public:

	static auto factory(Settings* settings) {
		return new AdjustmentCalculator(settings);
	}

	AdjustmentCalculator(Settings* settings);
	void AddMatrix(int rows, int cols);
	void SetBatchSize(int batchSize);
	void AddWeightAdjustment(int matrixIndex, PyNet::Models::Matrix* adjustments);
	void AddBiasAdjustment(int matrixIndex, double adjustment);
	PyNet::Models::Matrix* GetWeightAdjustment(int matrixIndex);
	PyNet::Models::Vector* GetBiasAdjustment(int matrixIndex);
	void SetNewBatch(bool newBatch);
};

