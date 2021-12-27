#include "NetworkTrainer.h"
#include <algorithm>

void NetworkTrainer::Backpropagate(std::vector<std::unique_ptr<Matrix>>& weightMatrices,
    std::vector<std::unique_ptr<Vector>>& layers, PyNet::Models::Vector& expectedLayer, std::shared_ptr<Vector> lossDerivative) {

    _dError_dLayerAbove = std::unique_ptr<Vector>(lossDerivative.get());
    _dError_dActivatedOutput = std::unique_ptr<Vector>(lossDerivative.get());

    for (size_t i = weightMatrices.size() - 1; i >= 0; i--) {
        
        auto dError_dBias = GetdError_dBias(layers[i + 1], i);
        _adjustmentCalculator->AddBiasAdjustment(i, dError_dBias);

        auto dError_dWeight = GetdError_dWeight(weightMatrices[i], layers[i-1], layers[i], i);
        _adjustmentCalculator->AddWeightAdjustment(i, std::move(dError_dWeight));
    }

    _logger->LogLine("Finished backpropagating.");

    _adjustmentCalculator->SetNewBatch(false);
}

void NetworkTrainer::UpdateWeights(std::vector<std::unique_ptr<Matrix>>& weightMatrices, std::vector<std::unique_ptr<Vector>>& biases, double learningRate) {
    _logger->LogLine("Updating weights...");

    for (int index = weightMatrices.size() - 1; index >= 0; index--) {

        auto bias = biases[index].get();
        auto weightMatrix = weightMatrices[index].get();
   
        auto biasAdjustmentMatrix = *_adjustmentCalculator->GetBiasAdjustment(index) * learningRate;
        auto biasAdjustmentVector = std::unique_ptr<Vector>(dynamic_cast<Vector*>(std::move(biasAdjustmentMatrix.get())));

        biases[index] = std::move(*bias - *biasAdjustmentVector);
        weightMatrices[index] = *weightMatrix - *(*_adjustmentCalculator->GetWeightAdjustment(index) * learningRate);
    }

    _adjustmentCalculator->SetNewBatch(true);
}

double NetworkTrainer::GetdError_dBias(std::unique_ptr<Vector>& outputLayer, int index) 
{
 /*   _logger->LogLine("Getting dError_dBias for layer: " + std::to_string(index));

    auto& activatedOutputLayer = outputLayer.CalculateActivationDerivative();
    return _dError_dActivatedOutput | activatedOutputLayer;*/
    return 0;
}

std::unique_ptr<Matrix> NetworkTrainer::GetdError_dWeight(std::unique_ptr<Matrix>& layerAboveMatrix, std::unique_ptr<Vector>& inputLayer, std::unique_ptr<Vector>& outputLayer, int index) {

    _logger->LogLine("Backpropagating for weight matrix: " + std::to_string(index));

    auto layerAboveTranspose = ~*layerAboveMatrix;
    auto dError_dActivatedOutput = std::unique_ptr<Vector>(dynamic_cast<Vector*>((*layerAboveTranspose * *_dError_dLayerAbove).get()));
    auto dError_dOutput = *dError_dActivatedOutput ^ *outputLayer->CalculateActivationDerivative();

    auto inputLayerTranspose = ~*inputLayer;
    auto dError_dWeight = *dError_dOutput * *inputLayerTranspose;

    _dError_dLayerAbove.reset(dError_dOutput.get());

    return dError_dWeight;
}