#pragma once

#include "PyNet.Models/Matrix.h"
#include <memory>
#include <vector>
#include "PyNet.Models/Vector.h"
#include "NetworkTrainer.h"
#include "LayerPropagator.h"
#include "ILogger.h"

class PyNetwork
{
private:
	std::unique_ptr<LayerPropagator> _layerPropagator;
	std::shared_ptr<ILogger> _logger;
	std::shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
	std::unique_ptr<NetworkTrainer> _networkTrainer;
public:
	std::vector<std::unique_ptr<Models::Vector>> Layers;
	std::vector<std::unique_ptr<Matrix>> Weights;
	std::vector<double> Errors;
	int BatchSize;
	int BatchNumber;
	double LearningRate;
	int NumberOfExamples;
	int CurrentIteration;

	PyNetwork(int, std::shared_ptr<ILogger> logger);
	void AddLayer(int, ActivationFunctionType);
	void Run(double*, double*);
	double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate);
	//void AddAdjustment(int matrixIndex, int row, int col, double adjustment);
};
