#pragma once

#include "PyNet.Models/Matrix.h"
#include <memory>
#include <vector>
#include "PyNet.Models/Vector.h"
#include "NetworkTrainer.h"
#include "Context.h"
#include "LayerPropagator.h"
#include "PyNet.Models/ILogger.h"

class PyNetwork
{
private:
	std::unique_ptr<LayerPropagator> _layerPropagator;
	std::shared_ptr<ILogger> _logger;
	std::shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
	std::unique_ptr<NetworkTrainer> _networkTrainer;
	std::unique_ptr<di::Context> _context;
	bool _cudaEnabled;
public:
	std::vector<std::unique_ptr<PyNet::Models::Vector>> Layers;
	std::vector<std::unique_ptr<PyNet::Models::Matrix>> Weights;
	std::vector<std::unique_ptr<PyNet::Models::Vector>> Biases;
	std::vector<double> Errors;
	int BatchSize;
	int BatchNumber;
	double LearningRate;
	int NumberOfExamples;
	int CurrentIteration;

	PyNetwork(int, std::shared_ptr<ILogger> logger, bool cudaEnabled);
	void AddLayer(int, ActivationFunctionType);
	double* Run(double* input_layer);
	double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate);
};
