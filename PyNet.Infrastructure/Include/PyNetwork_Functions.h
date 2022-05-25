#pragma once
#include "Activations/Activation.h"
#include "Headers.h"

using namespace PyNet::Models;
using namespace PyNet::Infrastructure::Activations;

namespace PyNet::Infrastructure {
	extern "C" {

		EXPORT void* PyNetwork_Initialise(bool log, bool cudaEnabled);

		EXPORT void PyNetwork_Destruct(void* input);

		EXPORT void PyNetwork_AddInputLayer(void* input, int rows, int cols);

		EXPORT void PyNetwork_AddDenseLayer(void* input, int count, ActivationFunctionType activationFunctionType);

		EXPORT void PyNetwork_AddDropoutLayer(void* input, double rate, int rows, int cols);

		EXPORT void PyNetwork_AddConvolutionLayer(void* input, int filterSize, ActivationFunctionType activationFunctionType);

		EXPORT void PyNetwork_AddMaxPoolingLayer(void* input, int filterSize);

		EXPORT void PyNetwork_AddFlattenLayer(void* input);

		EXPORT void PyNetwork_AddSoftmaxLayer(void* input);

		EXPORT const double* PyNetwork_Run(void* input, double* inputLayer);

		EXPORT void PyNetwork_SetVariableLearning(void* input, double errorThreshold, double lrDecrease, double lrIncrease);

		EXPORT void PyNetwork_Train(void* input, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
			double momentum, int epochs);
	}
}