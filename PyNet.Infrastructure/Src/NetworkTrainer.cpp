#include "NetworkTrainer.h"
#include "PyNet.Models/Logistic.h"

using namespace Models;

NetworkTrainer::NetworkTrainer(std::shared_ptr<ILogger> logger, std::shared_ptr<AdjustmentCalculator> adjustmentCalculator) {
    _logger = logger;
    _adjustmentCalculator = adjustmentCalculator;
}

double NetworkTrainer::TrainNetwork(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers, Models::Vector* expectedLayer) {

    double error = CalculateErrorDerivativeForFinalLayer(layers[layers.size() - 1], expectedLayer);
    GetAdjustments(weightMatrices, layers);
    _adjustmentCalculator->IncrementBatchNumber();

    _logger->LogMessage("I am returning: ");
    _logger->LogNumber(error);
    _logger->LogNewline();

    return error;
}

double NetworkTrainer::CalculateErrorDerivativeForFinalLayer(Models::Vector* finalLayer, Models::Vector* expectedLayer) {

    _logger->LogMessage("Expected layer is:");
    _logger->LogDoubleArray(expectedLayer->GetAddress(1), expectedLayer->Rows);
    double error = 0;
    for (int b = 0; b < finalLayer->Rows; b++) {
        dError_dLayerAbove.push_back(-(expectedLayer->GetValue(b) - finalLayer->GetValue(b)));
        _logger->LogMessage("Expected value: ");
        _logger->LogNumber(expectedLayer->GetValue(b));
        _logger->LogNewline();
        _logger->LogMessage("Actual value: ");
        _logger->LogNumber(finalLayer->GetValue(b));
        _logger->LogNewline();
        error += 0.5 * (expectedLayer->GetValue(b) - finalLayer->GetValue(b)) * (expectedLayer->GetValue(b) - finalLayer->GetValue(b));
        _logger->LogMessage("Temp error is ");
        _logger->LogNumber(error);
        _logger->LogNewline();
    }
    _logger->LogLine("Calculated derivatives for final layer.");
    _logger->LogMessage("Error is: ");
    _logger->LogNumber(error);
    _logger->LogNewline();

    return error;
}

void NetworkTrainer::GetErrorDerivativeForOutputLayer(Matrix* weightMatrix, Models::Vector* outputLayer) {
    dError_dOutputCurrent.clear();
    _logger->LogLine("Calculating error derivative with respect to current output layer.");
    _logger->LogNumber(weightMatrix->Cols);
    _logger->LogNewline();
    for (auto col = 0; col < weightMatrix->Cols; col++) {
        dError_dOutputCurrent.push_back(0);
        for (auto row = 0; row < weightMatrix->Rows; row++) {
            dError_dOutputCurrent[col] += dError_dLayerAbove[row] * outputLayer->CalculateActivationDerivative(outputLayer->GetValue(row)) * weightMatrix->GetValue(row, col);
        }
    }

    _logger->LogMessage("dError_dOutputCurrent: ");
    _logger->LogDoubleArray(dError_dOutputCurrent.data(), static_cast<int>(dError_dOutputCurrent.size()));
}

void NetworkTrainer::UpdateWeights(std::vector<Matrix*> weightMatrices, double learningRate) {
    _logger->LogLine("Updating weights...");
    for (int weightMatrixIndex = static_cast<int>(weightMatrices.size() - 1); weightMatrixIndex >= 0; weightMatrixIndex--) {
        Matrix* weightMatrix = weightMatrices[weightMatrixIndex];

        for (int row = 0; row < weightMatrix->Rows; row++) {
            for (int col = 0; col < weightMatrix->Cols; col++) {
                double* wij = weightMatrix->GetAddress(row, col);
                *wij = *wij - learningRate * _adjustmentCalculator->GetAdjustment(weightMatrixIndex, row, col);
            }
        }
    }
}

void NetworkTrainer::UpdateErrorDerivativeForLayerAbove(int length) {

    dError_dLayerAbove.clear();
    dError_dLayerAbove = std::vector<double>(dError_dOutputCurrent.size());
    std::copy(&dError_dOutputCurrent[0], &dError_dOutputCurrent[dError_dOutputCurrent.size() -1], dError_dLayerAbove.begin());

    _logger->LogMessage("dError_dLayerAbove: ");
    _logger->LogDoubleArray(dError_dLayerAbove.data(), length);
    _logger->LogNewline();
}

void NetworkTrainer::GetAdjustmentsForWeightMatrix(Matrix* weightMatrix, Vector* inputLayer, Vector* outputLayer, int weightMatrixIndex) {
    _logger->LogMessage("Calcuating loop for weight matrix: ");
    _logger->LogNumber(weightMatrixIndex);
    _logger->LogNewline();

    GetErrorDerivativeForOutputLayer(weightMatrix, outputLayer);

    _logger->LogLine("Calculating adjustments.");

    for (auto row = 0; row < weightMatrix->Rows; row++) {
        for (auto col = 0; col < weightMatrix->Cols; col++) {

            double dActivation_dWeightIJ = inputLayer->GetValue(col);
            double daij = dError_dOutputCurrent[col] * outputLayer->CalculateActivationDerivative(outputLayer->GetValue(col)) * dActivation_dWeightIJ;
            _adjustmentCalculator->AddAdjustment(weightMatrixIndex, row, col, daij);
        }
    }

    UpdateErrorDerivativeForLayerAbove(weightMatrix->Cols);
}

void NetworkTrainer::GetAdjustments(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers) {
    for (int a = static_cast<int>(weightMatrices.size() - 1); a >= 0; a--) {
        GetAdjustmentsForWeightMatrix(weightMatrices[a], layers[a], layers[(size_t)a+1], a);
    }
}