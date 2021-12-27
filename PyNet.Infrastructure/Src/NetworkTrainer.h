#pragma once

#include <memory>
#include <vector>
#include "PyNet.Models/ILogger.h"
#include "PyNet.Models/Vector.h"
#include "PyNet.DI/Context.h"
#include "PyNet.Models/Loss.h"
#include "AdjustmentCalculator.h"

using namespace PyNet::Models;

class NetworkTrainer
{
private:
	std::unique_ptr<Vector> _dError_dLayerAbove;
	std::unique_ptr<Vector> _dError_dActivatedOutput;
	std::shared_ptr<ILogger> _logger;
	std::shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
	std::shared_ptr<Settings> _settings;
	std::shared_ptr<Loss> _loss;
	std::shared_ptr<PyNet::DI::Context> _context;

	std::unique_ptr<Matrix> GetdError_dWeight(std::unique_ptr<Matrix>& weightMatrix, std::unique_ptr<Vector>& inputLayer, std::unique_ptr<Vector>& outputLayer, int weightMatrixIndex);
	double GetdError_dBias(std::unique_ptr<Vector>& outputLayer, int index);

public:

	static auto factory(std::shared_ptr<ILogger> logger, std::shared_ptr<AdjustmentCalculator> adjustmentCalculator, std::shared_ptr<Settings> settings, 
		std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Loss> loss) {
		return new NetworkTrainer{ logger, adjustmentCalculator, settings, context, loss };
	}

	NetworkTrainer(std::shared_ptr<ILogger> logger, std::shared_ptr<AdjustmentCalculator> adjustmentCalculator, std::shared_ptr<Settings> settings,
		std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Loss> loss) : _logger(logger), _adjustmentCalculator(adjustmentCalculator), 
		_settings(settings), _context(context), _loss{ loss } {
			_dError_dLayerAbove = context->GetUnique<Vector>();
			_dError_dActivatedOutput = context->GetUnique<Vector>();
	}

	void Backpropagate(std::vector<std::unique_ptr<Matrix>>& weightMatrices, std::vector<std::unique_ptr<Vector>>& layers, PyNet::Models::Vector& expectedLayer, std::shared_ptr<PyNet::Models::Vector> lossDerivative);
	void UpdateWeights(std::vector<std::unique_ptr<Matrix>>& weightMatrices, std::vector<std::unique_ptr<Vector>>& biases, double learningRate);
};

