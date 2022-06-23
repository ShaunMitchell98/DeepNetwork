#include "PyNetwork_Functions.h"
#include "Settings.h"
#include "Intermediary.h"
#include "Startup.h"
#include "PyNetwork.h"
#include "AdjustmentCalculator.h"
#include "RunMode.h"
#include "Layers/InputLayer.h"
#include "Layers/DenseLayer.h"
#include "Layers/ConvolutionalLayer.h"
#include "Layers/MaxPoolingLayer.h"
#include "Layers/FlattenLayer.h"
#include "Activations/Logistic.h"
#include "Layers/SoftmaxLayer.h"
#include "NetworkRunner.h"
#include "NetworkTrainer.h"
#include "PyNet.Models/ILogger.h"
#include <memory>

using namespace PyNet::DI;
using namespace std;

namespace PyNet::Infrastructure {

	EXPORT void* PyNetwork_Initialise(LogLevel logLevel, bool cudaEnabled) {
		auto settings = make_shared<Settings>();
		settings->LogLevel = logLevel;
		settings->CudaEnabled = cudaEnabled;

		auto startup = make_unique<Startup>();
		auto contextBuilder = make_unique<ContextBuilder>();

		startup->RegisterServices(*contextBuilder, settings);

		auto context = contextBuilder->Build();
		auto intermediary = new Intermediary(context, settings);
		return intermediary;
	}

	EXPORT void PyNetwork_Destruct(void* input) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		context->MakeReferencesWeak();
		delete intermediary;
	}

	EXPORT void PyNetwork_AddInputLayer(void* input, int rows, int cols, double dropoutRate) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		auto inputLayer = context->GetUnique<InputLayer>();
		inputLayer->Initialise(rows, cols);
		inputLayer->DropoutRate = dropoutRate;
		pyNetwork->Layers.push_back(move(inputLayer));
	}

	EXPORT void PyNetwork_AddDenseLayer(void* input, int count, ActivationFunctionType activationFunctionType, double dropoutRate) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		auto cols = pyNetwork->Layers.back()->GetRows();

		auto denseLayer = context->GetUnique<DenseLayer>();
		denseLayer->Initialise(count, cols);
		denseLayer->DropoutRate = dropoutRate;

		if (activationFunctionType == ActivationFunctionType::Logistic) {
			auto logistic = context->GetUnique<Logistic>();
			logistic->Initialise(count, 1);
			denseLayer->SetActivation(move(logistic));
		}

		pyNetwork->Layers.push_back(move(denseLayer));
	}

	EXPORT void PyNetwork_AddConvolutionLayer(void* input, int filterSize, ActivationFunctionType activationFunctionType) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();

		auto pyNetwork = context->GetShared<PyNetwork>();

		auto rows = pyNetwork->Layers.back()->GetRows();
		auto cols = pyNetwork->Layers.back()->GetCols();
		
		auto convolutionalLayer = context->GetUnique<ConvolutionalLayer>();
		convolutionalLayer->Initialise(filterSize, rows, cols);
		pyNetwork->Layers.push_back(move(convolutionalLayer));
	}

	EXPORT void PyNetwork_AddMaxPoolingLayer(void* input, int filterSize) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();

		auto pyNetwork = context->GetShared<PyNetwork>();

		auto rows = pyNetwork->Layers.back()->GetRows();
		auto cols = pyNetwork->Layers.back()->GetCols();
		
		auto maxPoolingLayer = context->GetUnique<MaxPoolingLayer>();
		maxPoolingLayer->Initialise(filterSize, rows, cols);
		pyNetwork->Layers.push_back(move(maxPoolingLayer));
	}

	EXPORT void PyNetwork_AddFlattenLayer(void* input) {
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		auto rows = pyNetwork->Layers.back()->GetRows();
		auto cols = pyNetwork->Layers.back()->GetCols();
		
		auto flattenLayer = context->GetUnique<FlattenLayer>();
		flattenLayer->Initialise(rows, cols);
		pyNetwork->Layers.push_back(move(flattenLayer));
	}

	//EXPORT void PyNetwork_AddSoftmaxLayer(void* input) {
	//	auto intermediary = static_cast<Intermediary*>(input);
	//	auto context = intermediary->GetContext();

	//	auto pyNetwork = context->GetShared<PyNetwork>();
	//	auto softmaxLayer = context->GetUnique<SoftmaxLayer>();

	//	auto rows = pyNetwork->Layers.back()->GetRows();

	//	softmaxLayer->Initialise(rows, 1);
	//	pyNetwork->Layers.push_back(move(softmaxLayer));
	//}

	EXPORT const double* PyNetwork_Run(void* input, double* inputLayer) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto settings = context->GetShared<Settings>();
		settings->RunMode = RunMode::Running;
		auto networkRunner = context->GetShared<NetworkRunner>();

		auto pyNetwork = context->GetShared<PyNetwork>();
		auto inputMatrix = context->GetShared<Matrix>();
		inputMatrix->Initialise(pyNetwork->Layers.front()->GetRows(), pyNetwork->Layers.front()->GetCols());
		*inputMatrix = inputLayer;


		auto output = networkRunner->Run(inputMatrix);
		return output->GetAddress(1, 1);
	}

	EXPORT void PyNetwork_SetVariableLearning(void* input, double errorThreshold, double lrDecrease, double lrIncrease) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto vlSettings = context->GetShared<VariableLearningSettings>();
		vlSettings->ErrorThreshold = errorThreshold;
		vlSettings->LRDecrease = lrDecrease;
		vlSettings->LRIncrease = lrIncrease;
	}

	EXPORT void PyNetwork_Train(void* input, double** inputLayers, double** expectedOutputs, Settings* chosenSettings) {

		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto logger = context->GetShared<ILogger>();

		try 
		{
			auto settings = context->GetShared<Settings>();
			auto trainingState = context->GetShared<TrainingState>();

			settings->BaseLearningRate = chosenSettings->BaseLearningRate;
			settings->BatchSize = chosenSettings->BatchSize;
			settings->Epochs = chosenSettings->Epochs;
			settings->Momentum = chosenSettings->Momentum;
			settings->NumberOfExamples = chosenSettings->NumberOfExamples;
			settings->StartExampleNumber = chosenSettings->StartExampleNumber;

			settings->RunMode = RunMode::Training;
			trainingState->NewBatch = true;
			auto networkTrainer = context->GetShared<NetworkTrainer>();

			auto pyNetwork = context->GetShared<PyNetwork>();
			auto inputBaseMatrix = context->GetUnique<Matrix>();
			inputBaseMatrix->Initialise(pyNetwork->Layers.front()->GetRows(), pyNetwork->Layers.front()->GetCols());

			auto expectedOutputBaseMatrix = context->GetShared<Matrix>();
			expectedOutputBaseMatrix->Initialise(pyNetwork->Layers.back()->GetRows(), pyNetwork->Layers.back()->GetCols());

			auto trainingPairs = vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>>();

			for (auto i = 0; i < settings->NumberOfExamples; i++)
			{
				auto inputMatrix = shared_ptr<Matrix>(inputBaseMatrix->Copy().release());
				*inputMatrix = inputLayers[i];

				auto expectedOutputMatrix = shared_ptr<Matrix>(expectedOutputBaseMatrix->Copy().release());
				*expectedOutputMatrix = expectedOutputs[i];

				pair<shared_ptr<Matrix>, shared_ptr<Matrix>> pair;
				pair.first = inputMatrix;
				pair.second = expectedOutputMatrix;

				try {
				
					trainingPairs.push_back(pair);
				}
				catch (char* message) {
					auto a = 5;
				}
			}

			networkTrainer->TrainNetwork(trainingPairs);
		}
		catch (const char* message) 
		{
			logger->LogError(message);
			cout << message << endl;
		}
		
	}
}