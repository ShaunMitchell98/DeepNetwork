#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"
#include "Settings.h"

class AdjustmentCalculator
{
private:
	std::vector<std::unique_ptr<PyNet::Models::Matrix>> _weightAdjustments = std::vector<std::unique_ptr<PyNet::Models::Matrix>>();
	std::vector<std::unique_ptr<PyNet::Models::Vector>> _biasAdjustments = std::vector<std::unique_ptr<PyNet::Models::Vector>>();
	bool _newBatch = true;
	int _batchSize = 0;
	std::shared_ptr<Settings> _settings;
	std::shared_ptr<PyNet::DI::Context> _context;
	AdjustmentCalculator(std::shared_ptr<Settings> settings, std::shared_ptr<PyNet::DI::Context> context) : _settings(settings), _context(context) {}
public:

	static auto factory(std::shared_ptr<Settings> settings, std::shared_ptr<PyNet::DI::Context> context) {
		return new AdjustmentCalculator(settings, context);
	}

	void AddMatrix(int rows, int cols);
	void SetBatchSize(int batchSize);
	void AddWeightAdjustment(int matrixIndex, std::unique_ptr<PyNet::Models::Matrix> adjustments);
	void AddBiasAdjustment(int matrixIndex, double adjustment);
	std::unique_ptr<PyNet::Models::Matrix> GetWeightAdjustment(int matrixIndex);
	std::unique_ptr<PyNet::Models::Vector> GetBiasAdjustment(int matrixIndex);
	void SetNewBatch(bool newBatch);
};

