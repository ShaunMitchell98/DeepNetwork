#pragma once

#include "PyNet.Models/Activation.h"
#include "VariableLearningSettings.h"

namespace PyNet::Infrastructure {

	extern "C" {

		void* PyNetwork_New(bool log, bool cudaEnabled);
		int PyNetwork_Load(void* pyNetwork, const char* filePath);
		void PyNetwork_AddLayer(void* pyNetwork, int count, PyNet::Models::ActivationFunctionType activationFunctionType);
		double* PyNetwork_Run(void* pyNetwork, double* input_layer);
		void PyNetwork_SetVariableLearning(void* pyNetwork, VariableLearningSettings* vlSettings);
		void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
			double momentum, int epochs);
		void PyNetwork_Save(void* pyNetwork, const char* filePath);

	}
}
