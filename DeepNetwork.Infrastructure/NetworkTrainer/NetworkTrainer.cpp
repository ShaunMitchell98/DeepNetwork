#include "NetworkTrainer.h"
#include "../Activation Functions/logistic_function.h"

NetworkTrainer::NetworkTrainer(network network) {
    logger = std::make_unique<Logger>();

    for (int i = 0; i < network.weightMatrixCount; i++) {
        adjustmentsStorage.push_back(std::vector<float>(network.weights[i].rows * network.weights[i].cols));
    }
}

float NetworkTrainer::TrainNetwork(network network, matrix expectedLayer) {
    float error = CalculateErrorDerivativeForFinalLayer(network.layers[network.layerCount - 1], expectedLayer);
    GetAdjustments(network);
    UpdateWeights(network);

    return error;
}

float NetworkTrainer::CalculateErrorDerivativeForFinalLayer(matrix finalLayer, matrix expectedLayer) {

    float error = 0;
    for (int b = 0; b < finalLayer.rows; b++) {
        dError_dLayerAbove.push_back(-(expectedLayer.values[b] - finalLayer.values[b]) * calculate_logistic_derivative(finalLayer.values[b]));
        error += 0.5 * (expectedLayer.values[b] - finalLayer.values[b]) * (expectedLayer.values[b] - finalLayer.values[b]);
    }
    logger->LogLine("Calculated derivatives for final layer.");

    return error;
}

void NetworkTrainer::GetErrorDerivativeForOutputLayer(matrix weightMatrix, matrix outputLayer) {
    logger->LogLine("Calculating error derivative with respect to current output layer.");
    for (int j = 0; j < weightMatrix.cols; j++) {
        dError_dOutputCurrent.push_back(0);
        for (int i = 0; i < weightMatrix.rows; i++) {
            dError_dOutputCurrent[j] += dError_dLayerAbove[i] * calculate_logistic_derivative(outputLayer.values[i]) * weightMatrix.values[i * weightMatrix.cols + j];
        }
    }

    logger->LogMessage("dError_dOutputCurrent: ");
    logger->LogFloatArray(dError_dOutputCurrent.data(), weightMatrix.cols);
}

void NetworkTrainer::UpdateWeights(network network) {
    logger->LogLine("Updating weights...");
    for (int a = network.weightMatrixCount - 1; a >= 0; a--) {
        matrix weightMatrix = network.weights[a];

        for (int y = 0; y < weightMatrix.rows; y++) {
            for (int p = 0; p < weightMatrix.cols; p++) {
                //logger->LogLine("row, col: %d, %d", y, p);
                logger->LogMessage("row, col: ");
                logger->LogNumber(y);
                logger->LogMessageWithoutDate(", ");
                logger->LogNumber(p);
                logger->LogNewline();
                float* wij = &weightMatrix.values[weightMatrix.cols * y + p];
                *wij = *wij - 0.01 * adjustmentsStorage[a][weightMatrix.cols * y + p];
            }
        }
    }
}

void NetworkTrainer::UpdateErrorDerivativeForLayerAbove(int length) {

    dError_dLayerAbove = dError_dOutputCurrent;

    logger->LogMessage("dError_dLayerAbove: ");
    logger->LogFloatArray(dError_dLayerAbove.data(), length);
    logger->LogNewline();
}

void NetworkTrainer::GetAdjustmentsForLayer(network network, int a) {
    logger->LogLine("Calcuating loop for weight matrix: %d", a);

    matrix weightMatrix = network.weights[a];
    logger->LogLine("Weight Matrix: ");
    logger->LogMatrix(weightMatrix);

    matrix inputLayer = network.layers[a];
    logger->LogLine("Input Layer: ");
    logger->LogMatrix(inputLayer);

    matrix outputLayer = network.layers[a + 1];
    logger->LogLine("Output Layer: ");
    logger->LogMatrix(outputLayer);

    GetErrorDerivativeForOutputLayer(weightMatrix, outputLayer);

    logger->LogLine("Calculating adjustments.");

    for (int i = 0; i < weightMatrix.rows; i++) {
        for (int j = 0; j < weightMatrix.cols; j++) {

            //logger->LogLine("row, col: %d, %d", i, j);
            logger->LogMessage("row, col: ");
            logger->LogNumber(i);
            logger->LogMessageWithoutDate(", ");
            logger->LogNumber(j);
            logger->LogNewline();
            float dOutputCurrentJ_dWeightIJ = inputLayer.values[j];
            float daij = dError_dOutputCurrent[j] * dOutputCurrentJ_dWeightIJ;
            adjustmentsStorage[a].push_back(daij);
        }
    }

    UpdateErrorDerivativeForLayerAbove(weightMatrix.cols);
}

void NetworkTrainer::GetAdjustments(network network) {
    for (int a = network.weightMatrixCount - 1; a >= 0; a--) {
        GetAdjustmentsForLayer(network, a);
    }
}