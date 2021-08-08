#include "NetworkTrainer.h"
#include "../Activation Functions/logistic_function.h"

NetworkTrainer::NetworkTrainer(PyNetwork* network) {
    logger = std::make_unique<Logger>();
}

double NetworkTrainer::TrainNetwork(PyNetwork* network, Matrix* expectedLayer) {

    double error = CalculateErrorDerivativeForFinalLayer(network->Layers[network->Layers.size() - 1].get(), expectedLayer);
    GetAdjustments(network);

    if (network->BatchNumber == network->BatchSize) {
        UpdateWeights(network);
        //network.batchNumber = 1;
    }

    logger->LogMessage("I am returning: ");
    logger->LogNumber(error);
    logger->LogNewline();

    return error;
}

double NetworkTrainer::CalculateErrorDerivativeForFinalLayer(Matrix* finalLayer, Matrix* expectedLayer) {

    logger->LogMessage("Expected layer is:");
    logger->LogDoubleArray(expectedLayer->Values.data(), expectedLayer->Rows);
    double error = 0;
    for (int b = 0; b < finalLayer->Rows; b++) {
        dError_dLayerAbove.push_back(-(expectedLayer->Values[b] - finalLayer->Values[b]) * calculate_logistic_derivative(finalLayer->Values[b]));
        logger->LogMessage("Expected value: ");
        logger->LogNumber(expectedLayer->Values[b]);
        logger->LogNewline();
        logger->LogMessage("Actual value: ");
        logger->LogNumber(finalLayer->Values[b]);
        logger->LogNewline();
        error += 0.5 * (expectedLayer->Values[b] - finalLayer->Values[b]) * (expectedLayer->Values[b] - finalLayer->Values[b]);
        logger->LogMessage("Temp error is ");
        logger->LogNumber(error);
        logger->LogNewline();
    }
    logger->LogLine("Calculated derivatives for final layer.");
    logger->LogMessage("Error is: ");
    logger->LogNumber(error);
    logger->LogNewline();

    return error;
}

void NetworkTrainer::GetErrorDerivativeForOutputLayer(Matrix* weightMatrix, Matrix* outputLayer) {
    logger->LogLine("Calculating error derivative with respect to current output layer.");
    for (int j = 0; j < weightMatrix->Cols; j++) {
        dError_dOutputCurrent.push_back(0);
        for (int i = 0; i < weightMatrix->Rows; i++) {
            dError_dOutputCurrent[j] += dError_dLayerAbove[i] * calculate_logistic_derivative(outputLayer->Values[i]) * weightMatrix->Values[(size_t)i * weightMatrix->Cols + j];
        }
    }

    logger->LogMessage("dError_dOutputCurrent: ");
    logger->LogDoubleArray(dError_dOutputCurrent.data(), weightMatrix->Cols);
}

void NetworkTrainer::UpdateWeights(PyNetwork* network) {
    logger->LogLine("Updating weights...");
    for (int a = network->Weights.size() - 1; a >= 0; a--) {
        Matrix* weightMatrix = network->Weights[a].get();

        for (int y = 0; y < weightMatrix->Rows; y++) {
            for (int p = 0; p < weightMatrix->Cols; p++) {
                double* wij = &weightMatrix->Values[(size_t)weightMatrix->Cols * y + p];
                *wij = *wij - 0.1* network->Adjustments[a]->Values[(size_t)weightMatrix->Cols * y + p];
            }
        }
    }
}

void NetworkTrainer::UpdateErrorDerivativeForLayerAbove(int length) {

    dError_dLayerAbove = dError_dOutputCurrent;

    logger->LogMessage("dError_dLayerAbove: ");
    logger->LogDoubleArray(dError_dLayerAbove.data(), length);
    logger->LogNewline();
}

void NetworkTrainer::GetAdjustmentsForLayer(PyNetwork* network, int a) {
    logger->LogMessage("Calcuating loop for weight matrix: ");
    logger->LogNumber(a);
    logger->LogNewline();

    Matrix* weightMatrix = network->Weights[a].get();
    //logger->LogLine("Weight Matrix: ");
    //logger->LogMatrix(weightMatrix);

    Matrix* inputLayer = network->Layers[a].get();
    logger->LogLine("Input Layer: ");
    logger->LogMatrix(inputLayer);

    Matrix* outputLayer = network->Layers[(size_t)a + 1].get();
    logger->LogLine("Output Layer: ");
    logger->LogMatrix(outputLayer);

    GetErrorDerivativeForOutputLayer(weightMatrix, outputLayer);

    logger->LogLine("Calculating adjustments.");

    for (int i = 0; i < weightMatrix->Rows; i++) {
        for (int j = 0; j < weightMatrix->Cols; j++) {

            double dOutputCurrentJ_dWeightIJ = inputLayer->Values[j];
            double daij = dError_dOutputCurrent[j] * dOutputCurrentJ_dWeightIJ;
            network->AddAdjustment(a, weightMatrix->Cols * i + j, daij);
        }
    }

    UpdateErrorDerivativeForLayerAbove(weightMatrix->Cols);
}

void NetworkTrainer::GetAdjustments(PyNetwork* network) {
    for (int a = network->Weights.size() - 1; a >= 0; a--) {
        GetAdjustmentsForLayer(network, a);
    }
}