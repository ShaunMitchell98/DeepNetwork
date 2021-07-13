#pragma once

#include <memory>
#include <vector>
#include "../network.h"
#include "../Logging/logger.h"

class NetworkTrainer
{
private:
	std::vector<float> dError_dLayerAbove;
	std::vector<float> dError_dOutputCurrent;
	std::vector<std::vector<float>> adjustmentsStorage;
	std::unique_ptr<Logger> logger;

	float CalculateErrorDerivativeForFinalLayer(matrix finalLayer, matrix expectedLayer);
	void UpdateWeights(network network);
	void GetAdjustmentsForLayer(network network, int a);
	void GetAdjustments(network network);
	void UpdateErrorDerivativeForLayerAbove(int length);
	void GetErrorDerivativeForOutputLayer(matrix weightMatrix, matrix outputLayer);
public:
	NetworkTrainer(network network);
	float TrainNetwork(network network, matrix expectedLayer);
};

