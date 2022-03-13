module;
#include <vector>
#include <memory>
#include <string>
#include <fstream>
export module PyNet.Infrastructure:PyNetwork;

import :AdjustmentCalculator;
import :LayerPropagator;
import :TrainingAlgorithm;
import :LayerNormaliser;
import :Settings;
import :GradientCalculator;
import :VariableLearningSettings;
import :XmlReader;
import :XmlWriter;
import PyNet.Models;

using namespace std;
using namespace PyNet::Models;
using namespace PyNet::Infrastructure;

export class PyNetwork
{
private:
	shared_ptr<LayerPropagator> _layerPropagator;
	shared_ptr<ILogger> _logger;
	shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
	shared_ptr<TrainingAlgorithm> _trainingAlgorithm;
	shared_ptr<Settings> _settings;
	shared_ptr<PyNet::DI::Context> _context;
	shared_ptr<GradientCalculator> _gradientCalculator;
	shared_ptr<LayerNormaliser> _layerNormaliser;
	shared_ptr<Loss> _loss;
	VariableLearningSettings* _vlSettings = nullptr;
	vector<double> _losses = vector<double>();
	vector<unique_ptr<Vector>> _layers = vector<unique_ptr<Vector>>();
	vector<unique_ptr<Vector>> _biases = vector<unique_ptr<Vector>>();
	vector<unique_ptr<Matrix>> _weights = vector<unique_ptr<Matrix>>();

public:

	__declspec(dllexport) static auto factory(shared_ptr<ILogger> logger, shared_ptr<LayerPropagator> layerPropagator, shared_ptr<PyNet::DI::Context> context,
		shared_ptr<GradientCalculator> gradientCalculator, shared_ptr<AdjustmentCalculator> adjustmentCalculator,
		shared_ptr<TrainingAlgorithm> trainingAlgorithm, shared_ptr<Settings> settings, shared_ptr<Loss> loss,
		shared_ptr<LayerNormaliser> layerNormaliser) {
		return new PyNetwork{ logger, layerPropagator, context, gradientCalculator, adjustmentCalculator, trainingAlgorithm, settings, loss, layerNormaliser };
	}

	PyNetwork(shared_ptr<ILogger> logger, shared_ptr<LayerPropagator> layerPropagator, shared_ptr<PyNet::DI::Context> context, shared_ptr<GradientCalculator> gradientCalculator,
		shared_ptr<AdjustmentCalculator> adjustmentCalculator, shared_ptr<TrainingAlgorithm> trainingAlgorithm, shared_ptr<Settings> settings, shared_ptr<Loss> loss,
		shared_ptr<LayerNormaliser> layerNormaliser) :
		_logger{ logger }, _layerPropagator{ layerPropagator }, _context{ context }, _gradientCalculator{ gradientCalculator }, _adjustmentCalculator{ adjustmentCalculator },
		_trainingAlgorithm{ trainingAlgorithm }, _settings{ settings }, _loss{ loss }, _layerNormaliser{ layerNormaliser } {}

	__declspec(dllexport) int Load(const char* filePath) {

		auto reader = XmlReader::Create(filePath);

		if (reader->FindNode("Configuration")) {
			if (reader->FindNode("Weights")) {
				while (reader->FindNode("Weight")) {
					auto weightMatrix = _context->GetUnique<Matrix>();
					weightMatrix->Load(reader->ReadContent());
					_adjustmentCalculator->AddMatrix(weightMatrix->GetRows(), weightMatrix->GetCols());
					_weights.push_back(move(weightMatrix));
				}
			}

			if (reader->FindNode("Biases")) {
				while (reader->FindNode("Bias")) {
					auto biasVector = _context->GetUnique<Vector>();
					biasVector->Load(reader->ReadContent());
					_biases.push_back(move(biasVector));
				}
			}
		}

		auto layer = _context->GetUnique<Vector>();
		layer->Initialise(_weights[0]->GetCols(), false);
		_layers.push_back(move(layer));

		for (auto& m : _weights) {
			auto layer = _context->GetUnique<Vector>();
			layer->Initialise(m->GetRows(), false);
			_layers.push_back(move(layer));
		}

		return _layers[_layers.size() - 1]->GetRows();
	}

	__declspec(dllexport) void AddLayer(int rows) {
		if (_layers.empty()) {
			auto layer = move(_context->GetUnique<Vector>());
			layer->Initialise(rows, false);
			_layers.push_back(move(layer));
			return;
		}

		auto cols = _layers[_layers.size() - 1]->GetRows();

		auto layer = move(_context->GetUnique<Vector>());
		layer->Initialise(rows, false);
		_layers.push_back(move(layer));

		auto weightMatrix = move(_context->GetUnique<Matrix>());
		weightMatrix->Initialise(rows, cols, true);
		_weights.push_back(move(weightMatrix));

		auto biasVector = move(_context->GetUnique<Vector>());
		biasVector->Initialise(rows, true);

		_biases.push_back(move(biasVector));
		_adjustmentCalculator->AddMatrix(rows, cols);
	}

	__declspec(dllexport) double* Run(double* input_layer) {

		_layers[0]->Set(_layers[0]->GetRows(), input_layer);

		for (size_t i = 0; i < _weights.size(); i++) {
			_layerPropagator->PropagateLayer(*_weights[i], *_layers[i], *_biases[i], *_layers[i + 1]);
		}

		_layerNormaliser->NormaliseLayer(*_layers[_layers.size() - 1]);

		return _layers[_layers.size() - 1]->GetAddress(0);
	}

	__declspec(dllexport) double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate,
		double momentum, int epochs) {
		auto batchNumber = 1;
		auto currentIteration = 1;
		auto totalLossForCurrentEpoch = 0.0;
		auto learningRate = baseLearningRate;

		auto expectedVector = _context->GetUnique<Vector>();

		_adjustmentCalculator->SetBatchSize(batchSize);
		_adjustmentCalculator->SetMomentum(momentum);

		try {

			for (auto epoch = 0; epoch < epochs; epoch++) {
				for (auto i = 0; i < numberOfExamples; i++) {

					Run(inputLayers[i]);

					expectedVector->Set(_layers[_layers.size() - 1]->GetRows(), expectedOutputs[i]);

					auto loss = _loss->CalculateLoss(*expectedVector, *_layers[_layers.size() - 1]);

					totalLossForCurrentEpoch += loss;
					_logger->LogLine("The loss is: " + to_string(loss));
					_losses.push_back(loss);

					auto lossDerivative = _loss->CalculateDerivative(*expectedVector, *_layers[_layers.size() - 1]);

					_gradientCalculator->CalculateGradients(_weights, _layers, *expectedVector, *lossDerivative);

					if (batchNumber == batchSize) {

						_logger->LogLine("The learning rate is: " + to_string(learningRate));
						_trainingAlgorithm->UpdateWeights(_weights, _biases, learningRate, false);

						printf("Iteration %d, Error is %f\n", i, loss);
						batchNumber = 1;
					}
					else {
						batchNumber++;
					}

					currentIteration++;
				}

				if (_vlSettings != nullptr) {

					auto adjustedTotalLossForCurrentEpoch = 0.0;
					for (auto j = 0; j < numberOfExamples; j++) {
						Run(inputLayers[j]);
						expectedVector->Set(_layers[_layers.size() - 1]->GetRows(), expectedOutputs[j]);
						adjustedTotalLossForCurrentEpoch += _loss->CalculateLoss(*expectedVector, *_layers[_layers.size() - 1]);
					}

					if (adjustedTotalLossForCurrentEpoch > (1 + _vlSettings->ErrorThreshold) * totalLossForCurrentEpoch) {
						learningRate = learningRate * _vlSettings->LRDecrease;
						_trainingAlgorithm->UpdateWeights(_weights, _biases, learningRate, true);
						_adjustmentCalculator->SetMomentum(0);
					}
					else if (adjustedTotalLossForCurrentEpoch > totalLossForCurrentEpoch) {
						_adjustmentCalculator->SetMomentum(momentum);
					}
					else {
						learningRate = learningRate * _vlSettings->LRIncrease;
						_adjustmentCalculator->SetMomentum(momentum);
					}
				}

				totalLossForCurrentEpoch = 0.0;
			}

		}
		catch (const char* message) {
			_logger->LogLine(message);
		}

		return _losses.data();
	}

	__declspec(dllexport) void SetVLSettings(VariableLearningSettings* vlSettings) {
		_vlSettings = vlSettings;
	}

	__declspec(dllexport) void Save(const char* filePath) {
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
};