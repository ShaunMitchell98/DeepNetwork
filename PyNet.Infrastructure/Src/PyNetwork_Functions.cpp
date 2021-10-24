#include "../Include/PyNetwork_Functions.h"

#include "PyNetwork.h"
#include "Logger.h"
#include "Settings.h"

extern "C" {

	void* PyNetwork_New(int count, bool log, bool cudaEnabled) { 
		auto context = new di::ContextTmpl<Logger>();
		Settings* settings = new Settings();
		settings->CudaEnabled = cudaEnabled;
		settings->LoggingEnabled = log;
		context->addInstance<Settings>(settings, true);
		return context->get<PyNetwork>();
	}

	void PyNetwork_AddLayer(void* pyNetwork, int count, ActivationFunctions::ActivationFunctionType activationFunctionType) {
		((PyNetwork*)pyNetwork)->AddLayer(count, activationFunctionType); 
	}

	double* PyNetwork_Run(void* pyNetwork, double* input_layer) { 
		return ((PyNetwork*)pyNetwork)->Run(input_layer);
	}

	void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
		((PyNetwork*)pyNetwork)->Train(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate);
	}
}
