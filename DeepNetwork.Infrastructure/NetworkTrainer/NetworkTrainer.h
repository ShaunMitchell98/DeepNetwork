#pragma once

#include <memory>
#include <vector>
#include "../Logging/ILogger.h"
#include "../Models/Vector.h"
#include "../Adjustments/AdjustmentCalculator.h"

using namespace Models;

class NetworkTrainer
{
private:
	std::vector<double> dError_dLayerAbove;
	std::vector<double> dError_dOutputCurrent;
	std::shared_ptr<ILogger> _logger;
	std::shared_ptr<AdjustmentCalculator> _adjustmentCalculator;

	double CalculateErrorDerivativeForFinalLayer(Models::Vector* finalLayer, Models::Vector* expectedLayer);
	void GetAdjustmentsForWeightMatrix(Matrix* weightMatrix, Vector* inputLayer, Vector* outputLayer, int weightMatrixIndex);
	void GetAdjustments(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers);
	void UpdateErrorDerivativeForLayerAbove(int length);
	void GetErrorDerivativeForOutputLayer(Matrix* weightMatrix, Models::Vector* outputLayer);
public:
	NetworkTrainer(std::shared_ptr<ILogger> logger, std::shared_ptr<AdjustmentCalculator> adjustmentCalculator);
	double TrainNetwork(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers, Models::Vector* expectedLayer);
	void UpdateWeights(std::vector<Matrix*> weightMatrices, double learningRate);
};

