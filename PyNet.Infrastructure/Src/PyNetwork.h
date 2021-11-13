#pragma once

#include "PyNet.Models/Matrix.h"
#include <memory>
#include <vector>
#include "PyNet.Models/Vector.h"
#include "NetworkTrainer.h"
#include "PyNet.Models/Context.h"
#include "LayerPropagator.h"
#include "PyNet.Models/ILogger.h"
#include "Logger.h"

class PyNetwork
{
private:
	LayerPropagator& _layerPropagator;
	ILogger& _logger;
	AdjustmentCalculator& _adjustmentCalculator;
	NetworkTrainer& _networkTrainer;
	Settings& _settings;
	di::Context& _context;
public:
	std::vector<PyNet::Models::Vector> Layers = std::vector<PyNet::Models::Vector>();
	std::vector<PyNet::Models::Matrix> Weights = std::vector<PyNet::Models::Matrix>();
	std::vector<PyNet::Models::Vector> Biases = std::vector<PyNet::Models::Vector>();
	std::vector<double> Errors = std::vector<double>();
	int BatchSize = 0;
	int BatchNumber = 0;
	double LearningRate = 0;
	int NumberOfExamples = 0;
	int CurrentIteration = 0;

	static auto factory(ILogger& logger, LayerPropagator& layerPropagator, di::Context& context, AdjustmentCalculator& adjustmentCalculator,
		NetworkTrainer& networkTrainer, Settings& settings) {
		return new PyNetwork{ logger, layerPropagator, context, adjustmentCalculator, networkTrainer, settings };
	}

	PyNetwork(ILogger& logger, LayerPropagator& layerPropagator, di::Context& context,
		AdjustmentCalculator& adjustmentCalculator, NetworkTrainer& networkTrainer, Settings& settings) :
		_logger{ logger }, _layerPropagator{ layerPropagator }, _context{ context }, _adjustmentCalculator{ adjustmentCalculator }, _networkTrainer{ networkTrainer }, _settings{ settings } {}


	void AddInitialLayer(int rows);
	void AddLayer(int, ActivationFunctionType);
	double* Run(double* input_layer);
	double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate);
};
