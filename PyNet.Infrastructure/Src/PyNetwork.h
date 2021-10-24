#pragma once

#include "PyNet.Models/Matrix.h"
#include <memory>
#include <vector>
#include "PyNet.Models/Vector.h"
#include "NetworkTrainer.h"
#include "Context.h"
#include "LayerPropagator.h"
#include "PyNet.Models/ILogger.h"
#include "Logger.h"

class PyNetwork
{
private:
	LayerPropagator* _layerPropagator;
	ILogger* _logger;
	AdjustmentCalculator* _adjustmentCalculator;
	NetworkTrainer* _networkTrainer;
	Settings* _settings;
	di::Context* _context;
public:
	std::vector<std::shared_ptr<PyNet::Models::Vector>> Layers;
	std::vector<std::shared_ptr<PyNet::Models::Matrix>> Weights;
	std::vector<std::shared_ptr<PyNet::Models::Vector>> Biases;
	std::vector<double> Errors;
	int BatchSize;
	int BatchNumber;
	double LearningRate;
	int NumberOfExamples;
	int CurrentIteration;

	static auto factory(ILogger* logger, LayerPropagator* layerPropagator, di::Context* context, AdjustmentCalculator* adjustmentCalculator,
		NetworkTrainer* networkTrainer, Settings* settings) {
		return new PyNetwork(1, logger, layerPropagator, context, adjustmentCalculator, networkTrainer, settings);
	}

	PyNetwork(int rows, ILogger* logger, LayerPropagator* layerPropagator, di::Context* context,
		AdjustmentCalculator* adjustmentCalculator, NetworkTrainer* networkTrainer, Settings* settings);

	// Add a layer to the network 
	void AddLayer(int, ActivationFunctionType);

	double* Run(double* input_layer);
	double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate);
};
