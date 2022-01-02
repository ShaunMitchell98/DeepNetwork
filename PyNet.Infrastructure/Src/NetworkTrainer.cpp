#include "NetworkTrainer.h"
#include <algorithm>

void NetworkTrainer::Backpropagate(std::vector<std::unique_ptr<Matrix>>& weightMatrices,
    std::vector<std::unique_ptr<Vector>>& layers, PyNet::Models::Vector& expectedLayer, std::shared_ptr<Vector> lossDerivative) {

    auto dActivatedLayerAbove_dLayerAbove = layers[layers.size() - 1]->CalculateActivationDerivative();
    _dError_dLayerAbove = *lossDerivative ^ *dActivatedLayerAbove_dLayerAbove;

    auto dError_dWeight = *_dError_dLayerAbove * *~*layers[layers.size() - 2];
    _logger->LogLine("dError_dWeight for final layer is: " + dError_dWeight->ToString());

    auto dError_dBias = *_dError_dLayerAbove | *dActivatedLayerAbove_dLayerAbove;
    _logger->LogLine("dError_dBias for final layer is: " + std::to_string(dError_dBias));

    _adjustmentCalculator->AddWeightAdjustment(weightMatrices.size() - 1, std::move(dError_dWeight));
    _adjustmentCalculator->AddBiasAdjustment(weightMatrices.size() - 1, dError_dBias);

    for (int i = weightMatrices.size() - 2; i >= 0; i--) {
        
        _logger->LogLine("Backpropagating for weight matrix: " + std::to_string(i));
        auto dError_dWeight = GetdError_dWeight(weightMatrices[i+1], layers[i], layers[(size_t)i+1]);
        _adjustmentCalculator->AddWeightAdjustment(i, std::move(dError_dWeight));

        _logger->LogLine("Getting dError_dBias for layer: " + std::to_string(i));
        auto dError_dBias = GetdError_dBias(layers[(size_t)i + 1]);
        _adjustmentCalculator->AddBiasAdjustment(i, dError_dBias);
    }

    _logger->LogLine("Finished backpropagating.");

    _adjustmentCalculator->SetNewBatch(false);
}

void NetworkTrainer::UpdateWeights(std::vector<std::unique_ptr<Matrix>>& weightMatrices, std::vector<std::unique_ptr<Vector>>& biases, double learningRate, bool reverse) {
    _logger->LogLine("Updating weights...");

    auto biasAdjustmentVector = _context->GetUnique<Vector>();

    for (int index = weightMatrices.size() - 1; index >= 0; index--) {
    
        auto biasAdjustmentMatrix = *_adjustmentCalculator->GetBiasAdjustment(index) * learningRate;
        biasAdjustmentVector->Set(biasAdjustmentMatrix->GetRows(), biasAdjustmentMatrix->GetValues().data());

        if (reverse) {
            biases[index] = *biases[index] + *biasAdjustmentVector;
            weightMatrices[index] = *weightMatrices[index] + *(*_adjustmentCalculator->GetWeightAdjustment(index) * learningRate);
        }
        else {
            biases[index] = *biases[index] - *biasAdjustmentVector;
            weightMatrices[index] = *weightMatrices[index] - *(*_adjustmentCalculator->GetWeightAdjustment(index) * learningRate);
        }
    }

    _adjustmentCalculator->SetNewBatch(true);
}

double NetworkTrainer::GetdError_dBias(std::unique_ptr<Vector>& outputLayer) 
{
    auto outputLayerDerivative = outputLayer->CalculateActivationDerivative();
    _logger->LogLine("Output layer derivative is: " + outputLayerDerivative->ToString());

    _logger->LogLine("dError_dLayerAbove is: " + _dError_dLayerAbove->ToString());

    return *_dError_dLayerAbove | *outputLayerDerivative;
}

std::unique_ptr<Matrix> NetworkTrainer::GetdError_dWeight(std::unique_ptr<Matrix>& layerAboveMatrix, std::unique_ptr<Vector>& inputLayer, std::unique_ptr<Vector>& outputLayer) {

    auto layerAboveTranspose = ~*layerAboveMatrix;
    auto dError_dActivatedOutputMatrix = *layerAboveTranspose * *_dError_dLayerAbove;
    auto dError_dActivatedOutput = _context->GetUnique<Vector>();
    dError_dActivatedOutput->Set(dError_dActivatedOutputMatrix->GetRows(), dError_dActivatedOutputMatrix->GetValues().data());

    auto dError_dOutput = *dError_dActivatedOutput ^ *outputLayer->CalculateActivationDerivative();

    auto inputLayerTranspose = ~*inputLayer;
    auto dError_dWeight = *dError_dOutput * *inputLayerTranspose;

    _dError_dLayerAbove = std::move(dError_dOutput);

    return std::move(dError_dWeight);
}