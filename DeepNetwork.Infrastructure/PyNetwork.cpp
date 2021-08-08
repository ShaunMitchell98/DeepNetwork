#include "PyNetwork.h"
#include <memory>
#include "Forward Propagation/forward_propagate.h"
#include "Train Network/train_network.h"
#include "NormaliseLayer.h"
#include "Logging/logger.h"
#include <cstddef>

PyNetwork::PyNetwork(int rows) {
	Layers = std::vector<std::unique_ptr<Matrix>>();
	Weights = std::vector<std::unique_ptr<Matrix>>();
	Adjustments = std::vector<std::unique_ptr<Matrix>>();
	Errors = std::vector<double>();

	Layers.push_back(std::make_unique<Matrix>(rows, 1));

	BatchNumber = 0;
	BatchSize = 0;
}

void PyNetwork::AddLayer(int rows) {
	auto cols = Layers[Layers.size() - 1]->Rows;

	Layers.push_back(std::make_unique<Matrix>(rows, 1));
	Weights.push_back(std::make_unique<Matrix>(rows, cols));
	Adjustments.push_back(std::make_unique<Matrix>(rows, cols));
}

void PyNetwork::Run(double* input_layer) {
	Layers[0].reset(new Matrix(Layers[0]->Rows, Layers[0]->Cols, input_layer));

	for (auto i = 0; i < Weights.size(); i++) {
		forward_propagate_layer(Weights[i].get(), Layers[i].get(), Layers[(size_t)i + 1].get(), activation_function::logistic);
	}

	normalise_layer(Layers.back().get());
}

double* PyNetwork::Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize) {
	BatchSize = batchSize;
	BatchNumber = 1;

	for (auto i = 0; i < numberOfExamples; i++) {

		Run(inputLayers[i]);

		auto expectedMatrix = std::make_unique<Matrix>(Layers[Layers.size() - 1]->Rows, 1, expectedOutputs[i]);

		auto error = train_network(this, expectedMatrix.get());
		Errors.push_back(error);

		if (BatchNumber == BatchSize) {
			printf("Iteration %d, Error is %f\n", i, error);
			BatchNumber = 1;
		}
		else {
			BatchNumber++;
		}
	}

	return Errors.data();
}

void PyNetwork::AddAdjustment(int matrixIndex, int valueIndex, double adjustment) {
	double* currentAdjustment = &Adjustments[matrixIndex]->Values[valueIndex];

	if (BatchNumber == 1) {
		*currentAdjustment = adjustment;
	}
	else {
		*currentAdjustment += adjustment;
	}
}