#pragma once

#include <vector>
#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "Settings.h"

class AdjustmentCalculator
{
private:
	std::vector <PyNet::Models::Matrix> _weightAdjustments = std::vector<PyNet::Models::Matrix>();
	std::vector<PyNet::Models::Vector> _biasAdjustments = std::vector<PyNet::Models::Vector>();
	bool _newBatch = true;
	int _batchSize = 0;
	Settings& _settings;
	di::Context& _context;
public:

	static auto factory(Settings& settings, di::Context& context) {
		return new AdjustmentCalculator(settings, context);
	}

	AdjustmentCalculator(Settings& settings, di::Context& context) : _settings(settings), _context(context) {}
	void AddMatrix(int rows, int cols);
	void SetBatchSize(int batchSize);
	void AddWeightAdjustment(int matrixIndex, PyNet::Models::Matrix* adjustments);
	void AddBiasAdjustment(int matrixIndex, double adjustment);
	PyNet::Models::Matrix* GetWeightAdjustment(int matrixIndex);
	PyNet::Models::Vector* GetBiasAdjustment(int matrixIndex);
	void SetNewBatch(bool newBatch);
};

