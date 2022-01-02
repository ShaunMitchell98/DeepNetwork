#pragma once

#include "PyNet.Models/Activation.h"
#include "VariableLearningSettings.h"

namespace PyNet::Infrastructure {

	extern "C" {

		__declspec(dllexport) void* PyNetwork_New(bool log, bool cudaEnabled);
		__declspec(dllexport) int PyNetwork_Load(void* pyNetwork, const char* filePath);
		__declspec(dllexport) void PyNetwork_AddLayer(void* pyNetwork, int count, PyNet::Models::ActivationFunctionType activationFunctionType);
		__declspec(dllexport) double* PyNetwork_Run(void* pyNetwork, double* input_layer);
		__declspec(dllexport) void PyNetwork_SetVariableLearning(void* pyNetwork, VariableLearningSettings* vlSettings);
		__declspec(dllexport) void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
			double momentum, int epochs);
		__declspec(dllexport) void PyNetwork_Save(void* pyNetwork, const char* filePath);

	}
}
