module;
#include <memory>
export module PyNet.Infrastructure:PyNetworkFunctions;

import :Setup;
import :PyNetwork;
import :VariableLearningSettings;
import PyNet.Models;

export namespace PyNet::Infrastructure {
	extern "C" {

		__declspec(dllexport) void* PyNetwork_New(bool log, bool cudaEnabled) {
			auto context = GetContext(cudaEnabled, log);
			auto pyNetwork = context->GetUnique<PyNetwork>();
			auto unmanagedPyNetwork = pyNetwork.release();
			return unmanagedPyNetwork;
		}

		__declspec(dllexport) int PyNetwork_Load(void* pyNetwork, const char* filePath) {
			return ((PyNetwork*)pyNetwork)->Load(filePath);
		}

		__declspec(dllexport) void PyNetwork_AddLayer(void* pyNetwork, int count, PyNet::Models::ActivationFunctionType activationFunctionType) {
			((PyNetwork*)pyNetwork)->AddLayer(count);
		}

		__declspec(dllexport) double* PyNetwork_Run(void* pyNetwork, double* input_layer) {
			return ((PyNetwork*)pyNetwork)->Run(input_layer);
		}

		__declspec(dllexport) void PyNetwork_SetVariableLearning(void* pyNetwork, VariableLearningSettings* vlSettings) {
			((PyNetwork*)pyNetwork)->SetVLSettings(vlSettings);
		}

		__declspec(dllexport) void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
			double momentum, int epochs) {
			((PyNetwork*)pyNetwork)->Train(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate, momentum, epochs);
		}

		__declspec(dllexport) void PyNetwork_Save(void* pyNetwork, const char* filePath) {
			((PyNetwork*)pyNetwork)->Save(filePath);
		}
	}
}