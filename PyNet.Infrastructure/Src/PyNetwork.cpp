#include "PyNetwork.h"
#include <memory>
#include "LayerPropagator.h"
#include "NormaliseLayer.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>

PyNetwork::PyNetwork(int rows, std::shared_ptr<ILogger> logger) {
	Layers = std::vector<std::unique_ptr<PyNet::Models::Vector>>();
	Weights = std::vector<std::unique_ptr<Matrix>>();
	Biases = std::vector<std::unique_ptr<PyNet::Models::Vector>>();
	Errors = std::vector<double>();

	Layers.push_back(std::make_unique<PyNet::Models::Vector>(rows));

	BatchNumber = 0;
	BatchSize = 0;
	LearningRate = 0;
	NumberOfExamples = 0;
	CurrentIteration = 0;

	_logger = logger;
	_layerPropagator = std::make_unique<LayerPropagator>(_logger);
	_adjustmentCalculator = std::make_shared<AdjustmentCalculator>();
	_networkTrainer = std::make_unique<NetworkTrainer>(_logger, _adjustmentCalculator);
}

void PyNetwork::AddLayer(int rows, ActivationFunctionType activationFunctionType) {

	auto cols = Layers[Layers.size() - 1]->Rows;

	Layers.push_back(std::make_unique<PyNet::Models::Vector>(rows, activationFunctionType));
	Weights.push_back(std::make_unique<Matrix>(rows, cols));
	Biases.push_back(std::make_unique<PyNet::Models::Vector>(rows));
	_adjustmentCalculator->AddMatrix(rows, cols);
}

double* PyNetwork::Run(double* input_layer) {
	Layers[0].reset(new PyNet::Models::Vector(Layers[0]->Rows, input_layer, ActivationFunctionType::Logistic));

	for (auto i = 0; i < Weights.size(); i++) {
		_layerPropagator->PropagateLayer(Weights[i].get(), Layers[i].get(), Biases[i].get(), Layers[(size_t)(i + 1)].get());
	}

	normalise_layer(Layers[Layers.size() - 1].get(), _logger.get());

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

			auto expectedVector = std::make_unique<PyNet::Models::Vector>(Layers[Layers.size() - 1]->Rows, expectedOutputs[i], ActivationFunctionType::Logistic);

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