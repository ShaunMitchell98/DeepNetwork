#pragma once

#include "../Models/Matrix.h"
#include <memory>
#include <vector>
#include "../Models/Vector.h"
#include "../NetworkTrainer/NetworkTrainer.h"
#include "../Forward Propagation/LayerPropagator.h"

class PyNetwork
{
private:
	std::unique_ptr<LayerPropagator> _layerPropagator;
	std::shared_ptr<Logger> _logger;
public:
	std::vector<std::unique_ptr<Models::Vector>> Layers;
	std::vector<std::unique_ptr<Matrix>> Weights;
	std::vector<double> Errors;
	int BatchSize;
	int BatchNumber;
	double LearningRate;
	int NumberOfExamples;
	int CurrentIteration;

	PyNetwork(int);
	void AddLayer(int, ActivationFunctionType);
	void Run(double*);
	double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate);
	//void AddAdjustment(int matrixIndex, int row, int col, double adjustment);
};
