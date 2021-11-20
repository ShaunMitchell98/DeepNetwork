#include "PyNetwork.h"
#include "LayerPropagator.h"
#include "NormaliseLayer.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>

void PyNetwork::AddInitialLayer(int rows) {
	auto& layer = _context.get<PyNet::Models::Vector>();
	layer.Initialise(rows);
	Layers.push_back(layer);
}

void PyNetwork::AddLayer(int rows, ActivationFunctionType activationFunctionType) {

	auto cols = Layers[Layers.size() - 1].get().GetRows();

	auto& layer = _context.get<PyNet::Models::Vector>();
	layer.SetActivationFunction(activationFunctionType);
	Layers.push_back(layer);

	auto& weightMatrix = _context.get<Matrix>();
	weightMatrix.Initialise(rows, cols);
	Weights.push_back(weightMatrix);

	auto& biasVector = _context.get<Vector>();
	biasVector.SetActivationFunction(activationFunctionType);
	biasVector.Initialise(rows);

	Biases.push_back(biasVector);
	_adjustmentCalculator.AddMatrix(rows, cols);
}

double* PyNetwork::Run(double* input_layer) {
	Layers[0].get() = input_layer;

	for (size_t i = 0; i < Weights.size(); i++) {
		_layerPropagator.PropagateLayer(&Weights[i].get(), &Layers[i].get(), &Biases[i].get(), &Layers[i + 1].get());
	}

	normalise_layer(Layers[Layers.size() - 1], _logger);

	return Layers[Layers.size() - 1].get().GetAddress(0);
}

double* PyNetwork::Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
	BatchSize = batchSize;
	BatchNumber = 1;
	NumberOfExamples = numberOfExamples;
	CurrentIteration = 1;
	LearningRate = learningRate;

	_adjustmentCalculator.SetBatchSize(batchSize);

	try {
		for (auto i = 0; i < numberOfExamples; i++) {

			Run(inputLayers[i]);

			auto& expectedVector = _context.get<Vector>();

			expectedVector = expectedOutputs[i];
			expectedVector.SetActivationFunction(ActivationFunctionType::Logistic);

			auto weights = std::vector<Matrix*>();
			auto layers = std::vector<Vector*>();
			auto biases = std::vector<Vector*>();

			for (auto i = 0; i < this->Weights.size(); i++) {
				weights.push_back(&this->Weights[i].get());
			}

			for (auto j = 0; j < this->Layers.size(); j++) {
				layers.push_back(&this->Layers[j].get());
			}

			for (auto k = 0; k < this->Biases.size(); k++) {
				biases.push_back(&this->Biases[k].get());
			}

			auto error = _networkTrainer.TrainNetwork(weights, layers, &expectedVector);
			Errors.push_back(error);

			if (BatchNumber == BatchSize) {
				auto learningRate = this->LearningRate * (static_cast<double>(this->NumberOfExamples) / this->CurrentIteration);
				_logger.LogLine("The learning rate is: ");
				_logger.LogNumber(this->LearningRate);
				_logger.LogNewline();
				_networkTrainer.UpdateWeights(weights, biases, learningRate);

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
		_logger.LogLine(message);
	}

	return Errors.data();
}