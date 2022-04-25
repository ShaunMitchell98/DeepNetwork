#pragma once
#include "PyNet.Models/Activation.h"
#include "Headers.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure {
	extern "C" {

		EXPORT void* PyNetwork_Initialise(bool log, bool cudaEnabled);

		EXPORT void PyNetwork_Destruct(void* input);

		EXPORT void PyNetwork_AddLayer(void* input, int count, ActivationFunctionType activationFunctionType, double dropoutRate);

		EXPORT double* PyNetwork_Run(void* input, double* inputLayer);

		EXPORT void PyNetwork_SetVariableLearning(void* input, double errorThreshold, double lrDecrease, double lrIncrease);

		EXPORT double* PyNetwork_Train(void* input, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
			double momentum, int epochs);
	}
}