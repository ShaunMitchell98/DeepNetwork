#include "PyNetwork_Functions.h"
#include "Settings.h"
#include "Intermediary.h"
#include "Setup.h"
#include "PyNetwork.h"
#include "AdjustmentCalculator.h"
#include "RunMode.h"
#include "Layers/InputLayer.h"
#include "Layers/DenseLayer.h"
#include "Layers/ConvolutionalLayer.h"
#include "Layers/MaxPoolingLayer.h"
#include "Layers/FlattenLayer.h"
#include "Layers/DropoutLayer.h"
#include "Activations/Logistic.h"
#include "NetworkRunner.h"
#include "NetworkTrainer.h"
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

	EXPORT void PyNetwork_AddInputLayer(void* input, int rows, int cols) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		auto inputLayer = context->GetUnique<InputLayer>();
		inputLayer->Initialise(rows, cols);
		pyNetwork->Layers.push_back(move(inputLayer));
	}

	EXPORT void PyNetwork_AddDenseLayer(void* input, int count, ActivationFunctionType activationFunctionType) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		auto cols = pyNetwork->Layers.back()->GetRows();

		auto denseLayer = context->GetUnique<DenseLayer>();
		denseLayer->Initialise(count, cols);
		pyNetwork->Layers.push_back(move(denseLayer));

		if (activationFunctionType == ActivationFunctionType::Logistic) {
			auto logisticLayer = context->GetUnique<Logistic>();
			logisticLayer->Initialise(count, 1);
			pyNetwork->Layers.push_back(move(logisticLayer));
		}
	}

	EXPORT void PyNetwork_AddDropoutLayer(void* input, double rate, int rows, int cols) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		auto dropoutLayer = context->GetUnique<DropoutLayer>();
		dropoutLayer->Initialise(rate, rows, cols);
		pyNetwork->Layers.push_back(move(dropoutLayer));
	}

	EXPORT void PyNetwork_AddConvolutionLayer(void* input, int filterSize, ActivationFunctionType activationFunctionType) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();

		auto pyNetwork = context->GetShared<PyNetwork>();
		auto convolutionalLayer = context->GetUnique<ConvolutionalLayer>();
		convolutionalLayer->Initialise(filterSize);
		pyNetwork->Layers.push_back(move(convolutionalLayer));
	}

	EXPORT void PyNetwork_AddMaxPoolingLayer(void* input, int filterSize) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();

		auto pyNetwork = context->GetShared<PyNetwork>();
		auto maxPoolingLayer = context->GetUnique<MaxPoolingLayer>();
		maxPoolingLayer->Initialise(filterSize);
		pyNetwork->Layers.push_back(move(maxPoolingLayer));
	}

	EXPORT void PyNetwork_AddFlattenLayer(void* input) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();

		auto pyNetwork = context->GetShared<PyNetwork>();
		auto flattenLayer = context->GetUnique<FlattenLayer>();
		pyNetwork->Layers.push_back(move(flattenLayer));
	}

	EXPORT const double* PyNetwork_Run(void* input, double* inputLayer) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto settings = context->GetShared<Settings>();
		settings->RunMode = RunMode::Running;
		auto networkRunner = context->GetShared<NetworkRunner>();
		auto output = networkRunner->Run(inputLayer);
		return output->GetAddress(1, 1);
	}

	EXPORT void PyNetwork_SetVariableLearning(void* input, double errorThreshold, double lrDecrease, double lrIncrease) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto networkTrainer = context->GetShared<NetworkTrainer>();
		networkTrainer->SetVLSettings(errorThreshold, lrDecrease, lrIncrease);
	}

	EXPORT void PyNetwork_Train(void* input, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate,
		double momentum, int epochs) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto settings = context->GetShared<Settings>();
		settings->RunMode = RunMode::Training;
		settings->Momentum = momentum;
		settings->NewBatch = true;
		auto networkTrainer = context->GetShared<NetworkTrainer>();
		networkTrainer->TrainNetwork(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate, momentum, epochs);
	}
}