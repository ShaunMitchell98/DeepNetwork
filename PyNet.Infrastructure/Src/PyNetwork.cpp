#include "PyNetwork.h"
#include "LayerPropagator.h"
#include "NormaliseLayer.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>

void PyNetwork::AddInitialLayer(int rows) {
	auto layer = _context->GetUnique<PyNet::Models::Vector>();
	layer->Initialise(rows, false);
	_layers.push_back(std::move(layer));
}

void PyNetwork::AddLayer(int rows, ActivationFunctionType activationFunctionType) {

	auto cols = _layers[_layers.size() - 1]->GetRows();

	auto layer = _context->GetUnique<PyNet::Models::Vector>();
	layer->Initialise(rows, false);
	layer->SetActivationFunction(activationFunctionType);
	_layers.push_back(std::move(layer));

	auto weightMatrix = _context->GetUnique<Matrix>();
	weightMatrix->Initialise(rows, cols, true);
	_weights.push_back(std::move(weightMatrix));

	auto biasVector = _context->GetUnique<Vector>();
	biasVector->SetActivationFunction(activationFunctionType);
	biasVector->Initialise(rows, true);

	_biases.push_back(std::move(biasVector));
	_adjustmentCalculator->AddMatrix(rows, cols);
}

double* PyNetwork::Run(double* input_layer) {
	_layers[0]->Set(_layers[0]->GetRows(), input_layer);

	for (size_t i = 0; i < _weights.size(); i++) {
		_layerPropagator->PropagateLayer(*_weights[i], *_layers[i], *_biases[i], _layers[i + 1]);
	}

	normalise_layer(*_layers[_layers.size() - 1], *_logger);

	return _layers[_layers.size() - 1]->GetAddress(0);
}

double* PyNetwork::Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate) {

	auto batchNumber = 1;
	auto currentIteration = 1;

	_adjustmentCalculator->SetBatchSize(batchSize);

	try {
		for (auto i = 0; i < numberOfExamples; i++) {

			Run(inputLayers[i]);

			auto expectedVector = _context->GetUnique<Vector>();
			expectedVector->Set(_layers[_layers.size() - 1]->GetRows(), expectedOutputs[i]);
			expectedVector->SetActivationFunction(ActivationFunctionType::Logistic);

			auto loss = _loss->CalculateLoss(*expectedVector, *_layers[_layers.size() - 1]);
			_logger->LogLine("The loss is: " + std::to_string(loss));
			_losses.push_back(loss);

			auto lossDerivative = _loss->CalculateDerivative(*expectedVector, *_layers[_layers.size() - 1]);

			_networkTrainer->Backpropagate(_weights, _layers, *expectedVector, std::move(lossDerivative));

			if (batchNumber == batchSize) {
				auto learningRate = baseLearningRate * (static_cast<double>(numberOfExamples) / currentIteration);
				_logger->LogLine("The learning rate is: " + std::to_string(learningRate));
				_networkTrainer->UpdateWeights(_weights, _biases, learningRate);

				printf("Iteration %d, Error is %f\n", i, loss);
				batchNumber = 1;
			}
			else {
				batchNumber++;
			}

			currentIteration++;
		} 
	}
	catch (const char* message) {
		_logger->LogLine(message);
	}

	return _losses.data();
}