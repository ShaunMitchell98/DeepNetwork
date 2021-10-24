#include "PyNetwork.h"
#include "LayerPropagator.h"
#include "NormaliseLayer.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>

PyNetwork::PyNetwork(int rows, ILogger* logger, LayerPropagator* layerPropagator,
	di::Context* context, AdjustmentCalculator* adjustmentCalculator, NetworkTrainer* networkTrainer, Settings* settings)  {
	Layers = std::vector<std::shared_ptr<PyNet::Models::Vector>>();
	Weights = std::vector<std::shared_ptr<Matrix>>();
	Biases = std::vector<std::shared_ptr<PyNet::Models::Vector>>();
	Errors = std::vector<double>();

	Layers.push_back(std::make_shared<PyNet::Models::Vector>(rows, settings->CudaEnabled));

	BatchNumber = 0;
	BatchSize = 0;
	LearningRate = 0;
	NumberOfExamples = 0;
	CurrentIteration = 0;

	_layerPropagator = layerPropagator;
	_networkTrainer = networkTrainer;
	_logger = logger;
	_context = context;
	_adjustmentCalculator = adjustmentCalculator;
	_settings = settings;
}

void PyNetwork::AddLayer(int rows, ActivationFunctionType activationFunctionType) {

	auto cols = Layers[Layers.size() - 1]->Rows;

	Layers.push_back(std::make_shared<PyNet::Models::Vector>(rows, activationFunctionType, _settings->CudaEnabled));
	Weights.push_back(std::make_shared<Matrix>(rows, cols, _settings->CudaEnabled));
	Biases.push_back(std::make_shared<PyNet::Models::Vector>(rows, _settings->CudaEnabled));
	_adjustmentCalculator->AddMatrix(rows, cols);
}

double* PyNetwork::Run(double* input_layer) {
	Layers[0].reset(new PyNet::Models::Vector(Layers[0]->Rows, input_layer, ActivationFunctionType::Logistic, _settings->CudaEnabled));

	for (auto i = 0; i < Weights.size(); i++) {
		_layerPropagator->PropagateLayer(Weights[i].get(), Layers[i].get(), Biases[i].get(), Layers[(size_t)(i + 1)].get());
	}

	normalise_layer(Layers[Layers.size() - 1].get(), _logger);

	return Layers[Layers.size() - 1].get()->GetAddress(0);
}

double* PyNetwork::Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
	BatchSize = batchSize;
	BatchNumber = 1;
	NumberOfExamples = numberOfExamples;
	CurrentIteration = 1;
	LearningRate = learningRate;

	_adjustmentCalculator->SetBatchSize(batchSize);

	try {
		for (auto i = 0; i < numberOfExamples; i++) {

			Run(inputLayers[i]);

			auto expectedVector = std::make_shared<PyNet::Models::Vector>(Layers[Layers.size() - 1]->Rows, expectedOutputs[i], ActivationFunctionType::Logistic, 
				_settings->CudaEnabled);

			auto weights = std::vector<Matrix*>();
			auto layers = std::vector<Vector*>();
			auto biases = std::vector<Vector*>();

			for (auto i = 0; i < this->Weights.size(); i++) {
				weights.push_back(this->Weights[i].get());
			}

			for (auto j = 0; j < this->Layers.size(); j++) {
				layers.push_back(this->Layers[j].get());
			}

			for (auto k = 0; k < this->Biases.size(); k++) {
				biases.push_back(this->Biases[k].get());
			}

			auto error = _networkTrainer->TrainNetwork(weights, layers, expectedVector.get());
			Errors.push_back(error);

			if (BatchNumber == BatchSize) {
				//auto learningRate = this->LearningRate * (static_cast<double>(this->NumberOfExamples) / this->CurrentIteration);
				_logger->LogLine("The learning rate is: ");
				_logger->LogNumber(this->LearningRate);
				_logger->LogNewline();
				_networkTrainer->UpdateWeights(weights, biases, this->LearningRate);

				printf("Iteration %d, Error is %f\n", i, error);
				BatchNumber = 1;
			}
			else {
				BatchNumber++;
			}

			CurrentIteration++;
		} 
	}
	catch (const char* message) {
		_logger->LogLine(message);
	}

	return Errors.data();
}