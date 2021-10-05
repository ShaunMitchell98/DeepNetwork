#pragma once

#include "PyNet.Models/Activation.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void* PyNetwork_New(int count, bool log);
	EXPORT_SYMBOL void PyNetwork_AddLayer(void* pyNetwork, int count, ActivationFunctionType activationFunctionType);
	EXPORT_SYMBOL double* PyNetwork_Run(void* pyNetwork, double* input_layer);
	EXPORT_SYMBOL void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate);

#undef EXPORT_SYMBOL
}