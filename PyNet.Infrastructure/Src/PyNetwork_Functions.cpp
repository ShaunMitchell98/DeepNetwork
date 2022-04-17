#include "PyNetwork_Functions.h"
#include "Settings.h"
#include "Intermediary.h"
#include "Setup.h"
#include "PyNetwork.h"
#include "AdjustmentCalculator.h"
#include <memory>

using namespace PyNet::DI;
using namespace std;

namespace PyNet::Infrastructure {

	EXPORT void* PyNetwork_Initialise(bool log, bool cudaEnabled) {
		auto settings = make_shared<Settings>();
		settings->LoggingEnabled = log;

		auto context = GetContext(settings, cudaEnabled);
		auto intermediary = new Intermediary(context, settings);
		return intermediary;
	}

	EXPORT void PyNetwork_Destruct(void* input) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		context->MakeReferencesWeak();
		delete intermediary;
	}

	EXPORT void PyNetwork_AddLayer(void* input, int count, ActivationFunctionType activationFunctionType) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();

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

	EXPORT double* PyNetwork_Run(void* input, double* inputLayer) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto networkRunner = context->GetShared<NetworkRunner>();
		return networkRunner->Run(inputLayer);
	}

	EXPORT void PyNetwork_SetVariableLearning(void* input, double errorThreshold, double lrDecrease, double lrIncrease) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto networkTrainer = context->GetShared<NetworkTrainer>();
		networkTrainer->SetVLSettings(errorThreshold, lrDecrease, lrIncrease);
	}

	EXPORT double* PyNetwork_Train(void* input, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
		double momentum, int epochs) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto networkTrainer = context->GetShared<NetworkTrainer>();
		return networkTrainer->TrainNetwork(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate, momentum, epochs);
	}
}