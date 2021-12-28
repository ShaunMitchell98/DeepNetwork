#include "PyNetwork_Functions.h"
#include "PyNetwork.h"
#include "Setup.h"

namespace PyNet::Infrastructure {
	extern "C" {

		void* PyNetwork_New(int count, bool log, bool cudaEnabled) {
			auto context = GetContext(cudaEnabled, log);
			auto pyNetwork = context->GetUnique<PyNetwork>();
			pyNetwork->AddInitialLayer(count);
			auto unmanagedPyNetwork = pyNetwork.release();
			return unmanagedPyNetwork;
		}

		void PyNetwork_AddLayer(void* pyNetwork, int count, PyNet::Models::ActivationFunctionType activationFunctionType) {
			((PyNetwork*)pyNetwork)->AddLayer(count);
		}

		double* PyNetwork_Run(void* pyNetwork, double* input_layer) {
			return ((PyNetwork*)pyNetwork)->Run(input_layer);
		}

		void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
			((PyNetwork*)pyNetwork)->Train(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate);
		}
	}
}

