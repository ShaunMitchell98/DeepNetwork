#include "NetworkTrainer.h"
#include <algorithm>

void NetworkTrainer::Backpropagate(std::vector<std::unique_ptr<Matrix>>& weightMatrices,
    std::vector<std::unique_ptr<Vector>>& layers, PyNet::Models::Vector& expectedLayer, std::shared_ptr<Vector> lossDerivative) {

    auto dActivatedLayerAbove_dLayerAbove = layers[layers.size() - 1]->CalculateActivationDerivative();
    _dError_dLayerAbove = *lossDerivative ^ *dActivatedLayerAbove_dLayerAbove;

    auto dError_dWeight = *_dError_dLayerAbove * *~*layers[layers.size() - 2];
    auto dError_dBias = *_dError_dLayerAbove | *layers[layers.size() - 1]->CalculateActivationDerivative();

    _adjustmentCalculator->AddWeightAdjustment(weightMatrices.size() - 1, std::move(dError_dWeight));
    _adjustmentCalculator->AddBiasAdjustment(weightMatrices.size() - 1, dError_dBias);

    for (int i = weightMatrices.size() - 2; i >= 0; i--) {
        
        _logger->LogLine("Getting dError_dBias for layer: " + std::to_string(i));
        auto dError_dBias = GetdError_dBias(layers[(size_t)i + 1]);
        _adjustmentCalculator->AddBiasAdjustment(i, dError_dBias);

        _logger->LogLine("Backpropagating for weight matrix: " + std::to_string(i));
        auto dError_dWeight = GetdError_dWeight(weightMatrices[i], layers[i], layers[(size_t)i+1]);
        _adjustmentCalculator->AddWeightAdjustment(i, std::move(dError_dWeight));
    }

    _logger->LogLine("Finished backpropagating.");

    _adjustmentCalculator->SetNewBatch(false);
}

void NetworkTrainer::UpdateWeights(std::vector<std::unique_ptr<Matrix>>& weightMatrices, std::vector<std::unique_ptr<Vector>>& biases, double learningRate) {
    _logger->LogLine("Updating weights...");

    for (int index = weightMatrices.size() - 1; index >= 0; index--) {
    
        auto biasAdjustmentMatrix = *_adjustmentCalculator->GetBiasAdjustment(index) * learningRate;
        auto biasAdjustmentVector = _context->GetUnique<Vector>();
        biasAdjustmentVector->Set(biasAdjustmentMatrix->GetRows(), biasAdjustmentMatrix->GetValues().data());

        biases[index] = *biases[index] - *biasAdjustmentVector;
        weightMatrices[index] = *weightMatrices[index] - *(*_adjustmentCalculator->GetWeightAdjustment(index) * learningRate);
    }

    _adjustmentCalculator->SetNewBatch(true);
}

double NetworkTrainer::GetdError_dBias(std::unique_ptr<Vector>& outputLayer) 
{
    return *_dError_dLayerAbove | *outputLayer->CalculateActivationDerivative();
}

std::unique_ptr<Matrix> NetworkTrainer::GetdError_dWeight(std::unique_ptr<Matrix>& layerAboveMatrix, std::unique_ptr<Vector>& inputLayer, std::unique_ptr<Vector>& outputLayer) {

    auto layerAboveTranspose = ~*layerAboveMatrix;
    auto dError_dActivatedOutput = std::unique_ptr<Vector>(dynamic_cast<Vector*>((*layerAboveTranspose * *_dError_dLayerAbove).get()));
    auto dError_dOutput = *dError_dActivatedOutput ^ *outputLayer->CalculateActivationDerivative();

    auto inputLayerTranspose = ~*inputLayer;
    auto dError_dWeight = *dError_dOutput * *inputLayerTranspose;

    _dError_dLayerAbove.reset(dError_dOutput.get());

    return std::move(dError_dWeight);
}