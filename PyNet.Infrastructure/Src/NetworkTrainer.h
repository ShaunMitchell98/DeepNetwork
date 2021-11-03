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
	Vector* dError_dActivatedLayerAbove;
	Vector* dError_dActivatedOutput;
	ILogger* _logger;
	AdjustmentCalculator* _adjustmentCalculator;
	Settings* _settings;

	double CalculateErrorDerivativeForFinalLayer(Vector* finalLayer, Vector* expectedLayer);
	void GetAdjustmentsForWeightMatrix(Matrix* weightMatrix, Vector* inputLayer, Vector* outputLayer, int weightMatrixIndex);
	void GetAdjustments(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers);
	void GetdError_dActivatedOutput(Matrix* weightMatrix, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* outputLayer);
public:

	static auto factory(ILogger* logger, AdjustmentCalculator* adjustmentCalculator, Settings* settings) {
		return new NetworkTrainer(logger, adjustmentCalculator, settings);
	}

	NetworkTrainer(ILogger* logger, AdjustmentCalculator* adjustmentCalculator, Settings* settings);
	double TrainNetwork(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers, PyNet::Models::Vector* expectedLayer);
	void UpdateWeights(std::vector<Matrix*> weightMatrices, std::vector<Vector*> biases, double learningRate);
};

