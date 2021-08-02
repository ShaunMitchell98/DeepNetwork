#pragma once

#include <memory>
#include <vector>
#include "../network.h"
#include "../Logging/logger.h"

class NetworkTrainer
{
private:
	std::vector<double> dError_dLayerAbove;
	std::vector<double> dError_dOutputCurrent;
	std::vector<std::vector<double>> adjustmentsStorage;
	std::unique_ptr<Logger> logger;

	double CalculateErrorDerivativeForFinalLayer(matrix finalLayer, matrix expectedLayer);
	void UpdateWeights(network network);
	void GetAdjustmentsForLayer(network network, int a);
	void GetAdjustments(network network);
	void UpdateErrorDerivativeForLayerAbove(int length);
	void GetErrorDerivativeForOutputLayer(matrix weightMatrix, matrix outputLayer);
public:
	NetworkTrainer(network network);
	double TrainNetwork(network network, matrix expectedLayer);
};

