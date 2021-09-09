#ifndef KERNEL_PYNETWORK_FUNCTIONS
#define KERNEL_PYNETWORK_FUNCTIONS

#include "../Src/PyNetwork.h"
#include "../Src/Logger.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL PyNetwork* PyNetwork_New(int count, bool log) { return new PyNetwork(count, std::make_shared<Logger>(log)); }
	EXPORT_SYMBOL void PyNetwork_AddLayer(PyNetwork* pyNetwork, int count, ActivationFunctionType activationFunctionType) { pyNetwork->AddLayer(count, activationFunctionType); }
	EXPORT_SYMBOL void PyNetwork_Run(PyNetwork* pyNetwork, double* input_layer, double* output_layer) { pyNetwork->Run(input_layer, output_layer); }
	EXPORT_SYMBOL void PyNetwork_Train(PyNetwork* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
		pyNetwork->Train(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate);
	}

#undef EXPORT_SYMBOL
}

#endif