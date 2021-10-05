#include "../Include/PyNetwork_Functions.h"

#include "PyNetwork.h"
#include "Logger.h"

extern "C" {

	void* PyNetwork_New(int count, bool log) { 
		return new PyNetwork(count, std::make_shared<Logger>(log));
	}

	void PyNetwork_AddLayer(void* pyNetwork, int count, ActivationFunctionType activationFunctionType) { 
		((PyNetwork*)pyNetwork)->AddLayer(count, activationFunctionType); 
	}

	double* PyNetwork_Run(void* pyNetwork, double* input_layer) { 
		return ((PyNetwork*)pyNetwork)->Run(input_layer);
	}

	void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
		((PyNetwork*)pyNetwork)->Train(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate);
	}
}
