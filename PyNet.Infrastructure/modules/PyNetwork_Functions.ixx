module;
#include <memory>
export module PyNet.Infrastructure:PyNetworkFunctions;

import :Setup;
import :PyNetwork;
import :VariableLearningSettings;
import :NetworkRunner;
import :NetworkTrainer;
import PyNet.Models;
import PyNet.DI;

using namespace PyNet::DI;
using namespace PyNet::Models;

export namespace PyNet::Infrastructure {
	extern "C" {

		__declspec(dllexport) void* PyNetwork_Initialise(bool log, bool cudaEnabled) {
			auto context = GetContext(cudaEnabled, log);
			return context.get();
		}

		__declspec(dllexport) void PyNetwork_AddLayer(void* input, int count, ActivationFunctionType activationFunctionType) {
			
			auto context = static_cast<Context*>(input);
			auto pyNetwork = context->GetShared<PyNetwork>();
			auto adjustmentCalculator = context->GetShared<AdjustmentCalculator>();

			if (pyNetwork->Layers.empty()) {
				auto layer = context->GetUnique<Vector>();
				layer->Initialise(count, false);
				pyNetwork->Layers.push_back(move(layer));
				return;
			}

			auto cols = pyNetwork->GetLastLayer().GetRows();

			auto layer = context->GetUnique<Vector>();
			layer->Initialise(count, false);
			pyNetwork->Layers.push_back(move(layer));

			auto weightMatrix = context->GetUnique<Matrix>();
			weightMatrix->Initialise(count, cols, true);
			pyNetwork->Weights.push_back(move(weightMatrix));

			auto biasVector = context->GetUnique<Vector>();
			biasVector->Initialise(count, true);

			pyNetwork->Biases.push_back(move(biasVector));
			adjustmentCalculator->AddMatrix(count, cols);
		}

		__declspec(dllexport) double* PyNetwork_Run(void* input, double* inputLayer) {

			auto context = static_cast<Context*>(input);
			auto networkRunner = context->GetShared<NetworkRunner>();
			return networkRunner->Run(inputLayer);
		}

		__declspec(dllexport) void PyNetwork_SetVariableLearning(void* input, VariableLearningSettings* vlSettings) {

			auto context = static_cast<Context*>(input);
			auto networkTrainer = context->GetShared<NetworkTrainer>();
			networkTrainer->SetVLSettings(vlSettings);
		}

		__declspec(dllexport) double* PyNetwork_Train(void* input, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
			double momentum, int epochs) {

			auto context = static_cast<Context*>(input);
			auto networkTrainer = context->GetShared<NetworkTrainer>();
			return networkTrainer->TrainNetwork(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate, momentum, epochs);
		}
	}
}