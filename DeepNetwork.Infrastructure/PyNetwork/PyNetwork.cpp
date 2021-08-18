#include "PyNetwork.h"
#include <memory>
#include "../Forward Propagation/LayerPropagator.h"
#include "../NormaliseLayer.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>

PyNetwork::PyNetwork(int rows) {
	Layers = std::vector<std::unique_ptr<Models::Vector>>();
	Weights = std::vector<std::unique_ptr<Matrix>>();
	Errors = std::vector<double>();

	Layers.push_back(std::make_unique<Models::Vector>(rows));

	BatchNumber = 0;
	BatchSize = 0;
	LearningRate = 0;
	NumberOfExamples = 0;
	CurrentIteration = 0;

	_logger = std::make_shared<Logger>();
	_layerPropagator = std::make_unique<LayerPropagator>(_logger);
}

void PyNetwork::AddLayer(int rows, ActivationFunctionType activationFunctionType) {
	auto cols = Layers[Layers.size() - 1]->Rows;

	Layers.push_back(std::make_unique<Models::Vector>(rows, activationFunctionType));
	Weights.push_back(std::make_unique<Matrix>(rows, cols));
}

void PyNetwork::Run(double* input_layer) {
	Layers[0].reset(new Models::Vector(Layers[0]->Rows, input_layer, ActivationFunctionType::Logistic));

	for (auto i = 0; i < Weights.size(); i++) {
		_layerPropagator->PropagateLayer(Weights[i].get(), Layers[i].get(), Layers[(size_t)(i + 1)].get());
	}

	normalise_layer(Layers[Layers.size() - 1].get());
}

double* PyNetwork::Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
	BatchSize = batchSize;
	BatchNumber = 1;
	NumberOfExamples = numberOfExamples;
	CurrentIteration = 1;
	auto networkTrainer = std::make_unique<NetworkTrainer>(_logger, batchSize, static_cast<int>(Layers.size()));

	for (auto i = 0; i < numberOfExamples; i++) {

		Run(inputLayers[i]);

		auto expectedVector = std::make_unique<Models::Vector>(Layers[Layers.size() - 1]->Rows, expectedOutputs[i], ActivationFunctionType::Logistic);

		auto weights = std::vector<Matrix*>();
		auto layers = std::vector<Vector*>();

		for (auto i = 0; i < this->Weights.size(); i++) {
			weights.push_back(this->Weights[i].get());
		}

		for (auto j = 0; j < this->Layers.size(); j++) {
			layers.push_back(this->Layers[j].get());
		}

		auto error = networkTrainer->TrainNetwork(weights, layers, expectedVector.get());
		Errors.push_back(error);

		if (BatchNumber == BatchSize) {
			auto learningRate = this->LearningRate * (static_cast<double>(this->NumberOfExamples) / this->CurrentIteration);
			networkTrainer->UpdateWeights(weights, learningRate);

			printf("Iteration %d, Error is %f\n", i, error);
			BatchNumber = 1;
		}
		else {
			BatchNumber++;
		}

		CurrentIteration++;
	}

	return Errors.data();
}