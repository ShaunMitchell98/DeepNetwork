#include "PyNetwork.h"
#include "LayerPropagator.h"
#include "NormaliseLayer.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>
#include "XmlWriter.h"
#include "XmlReader.h"

namespace PyNet::Infrastructure {

	int PyNetwork::Load(const char* filePath) {
		auto reader = XmlReader::Create(filePath);

		if (reader->FindNode("Configuration")) {
			if (reader->FindNode("Weights")) {
				while (reader->FindNode("Weight")) {
					auto weightMatrix = _context->GetUnique<Matrix>();
					weightMatrix->Load(reader->ReadContent());
					_adjustmentCalculator->AddMatrix(weightMatrix->GetRows(), weightMatrix->GetCols());
					_weights.push_back(std::move(weightMatrix));	
				}
			}

			if (reader->FindNode("Biases")) {
				while (reader->FindNode("Bias")) {
					auto biasVector = _context->GetUnique<Vector>();
					biasVector->Load(reader->ReadContent());
					_biases.push_back(std::move(biasVector));
				}
			}
		}

		auto layer = _context->GetUnique<Vector>();
		layer->Initialise(_weights[0]->GetCols(), false);
		_layers.push_back(std::move(layer));

		for (auto& m : _weights) {
			auto layer = _context->GetUnique<Vector>();
			layer->Initialise(m->GetRows(), false);
			_layers.push_back(std::move(layer));
		}

		return _layers[_layers.size() - 1]->GetRows();
	}

	void PyNetwork::AddLayer(int rows) {

		if (_layers.empty()) {
			auto layer = std::move(_context->GetUnique<PyNet::Models::Vector>());
			layer->Initialise(rows, false);
			_layers.push_back(std::move(layer));
			return;
		}

		auto cols = _layers[_layers.size() - 1]->GetRows();

		auto layer = std::move(_context->GetUnique<PyNet::Models::Vector>());
		layer->Initialise(rows, false);
		_layers.push_back(std::move(layer));

		auto weightMatrix = std::move(_context->GetUnique<Matrix>());
		weightMatrix->Initialise(rows, cols, true);
		_weights.push_back(std::move(weightMatrix));

		auto biasVector = std::move(_context->GetUnique<Vector>());
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

				auto expectedVector = std::move(_context->GetUnique<Vector>());
				expectedVector->Set(_layers[_layers.size() - 1]->GetRows(), expectedOutputs[i]);

				auto loss = _loss->CalculateLoss(*expectedVector, *_layers[_layers.size() - 1]);
				_logger->LogLine("The loss is: " + std::to_string(loss));
				_losses.push_back(loss);

				auto lossDerivative = _loss->CalculateDerivative(*expectedVector, *_layers[_layers.size() - 1]);

				_networkTrainer->Backpropagate(_weights, _layers, *expectedVector, std::move(lossDerivative));

				if (batchNumber == batchSize) {
					//auto learningRate = baseLearningRate * (static_cast<double>(numberOfExamples) / currentIteration);
					auto learningRate = baseLearningRate;
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

	void PyNetwork::Save(const char* filePath) {

		auto writer = XmlWriter::Create(filePath);

		writer->StartElement("Configuration");
		writer->StartElement("Weights");
		for (auto i = 0; i < _weights.size(); i++) {
			writer->StartElement("Weight");
			writer->WriteString(_weights[i]->ToString());
			writer->EndElement();
		}

		writer->EndElement();

		writer->StartElement("Biases");
		for (auto i = 0; i < _biases.size(); i++) {
			writer->StartElement("Bias");
			writer->WriteString(_biases[i]->ToString());
			writer->EndElement();
		}

		writer->EndElement();

		writer->EndElement();
	}
}