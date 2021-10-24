#pragma once

#include <memory>
#include <vector>
#include "PyNet.Models/ILogger.h"
#include "PyNet.Models/Vector.h"
#include "AdjustmentCalculator.h"

using namespace PyNet::Models;

class NetworkTrainer
{
private:
	std::vector<double> dError_dLayerAbove;
	std::vector<double> dError_dOutputCurrent;
	std::shared_ptr<ILogger> _logger;
	std::shared_ptr<AdjustmentCalculator> _adjustmentCalculator;

	double CalculateErrorDerivativeForFinalLayer(PyNet::Models::Vector* finalLayer, PyNet::Models::Vector* expectedLayer);
	void GetAdjustmentsForWeightMatrix(Matrix* weightMatrix, Vector* inputLayer, Vector* outputLayer, int weightMatrixIndex);
	void GetAdjustments(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers);
	void UpdateErrorDerivativeForLayerAbove();
	void GetErrorDerivativeForOutputLayer(Matrix* weightMatrix, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* outputLayer);
public:
	NetworkTrainer(std::shared_ptr<ILogger> logger, std::shared_ptr<AdjustmentCalculator> adjustmentCalculator);
	double TrainNetwork(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers, PyNet::Models::Vector* expectedLayer);
	void UpdateWeights(std::vector<Matrix*> weightMatrices, std::vector<Vector*> biases, double learningRate);
};

