#pragma once

#include "Matrix.h"
#include <memory>
#include <vector>

class PyNetwork
{
public:
	std::vector<std::unique_ptr<Matrix>> Layers;
	std::vector<std::unique_ptr<Matrix>> Weights;
	std::vector<std::unique_ptr<Matrix>> Adjustments;
	std::vector<double> Errors;
	int BatchSize;
	int BatchNumber;

	PyNetwork(int);
	void AddLayer(int);
	void Run(double*);
	double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize);
	void AddAdjustment(int matrixIndex, int valueIndex, double adjustment);
};
