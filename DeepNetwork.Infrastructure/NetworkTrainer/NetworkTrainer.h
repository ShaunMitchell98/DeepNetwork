#pragma once

#include <memory>
#include <vector>
#include "../PyNetwork.h"
#include "../Logging/logger.h"

class NetworkTrainer
{
private:
	std::vector<double> dError_dLayerAbove;
	std::vector<double> dError_dOutputCurrent;
	std::unique_ptr<Logger> logger;

	double CalculateErrorDerivativeForFinalLayer(Matrix* finalLayer, Matrix* expectedLayer);
	void UpdateWeights(PyNetwork* network);
	void GetAdjustmentsForLayer(PyNetwork* network, int a);
	void GetAdjustments(PyNetwork* network);
	void UpdateErrorDerivativeForLayerAbove(int length);
	void GetErrorDerivativeForOutputLayer(Matrix* weightMatrix, Matrix* outputLayer);
public:
	NetworkTrainer(PyNetwork* network);
	double TrainNetwork(PyNetwork* network, Matrix* expectedLayer);
};

