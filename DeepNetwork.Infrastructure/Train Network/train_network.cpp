#include "../Activation Functions/logistic_function.h"
#include "../Logging/log.h"
#include "train_network.h"

extern "C" {
    static float** initialise_adjustments_storage(network network) {
        log_line("Initialising adjustment storage...");

        float** adjustments = (float**)malloc(network.weightMatrixCount * sizeof(float*));

        for (int a = 0; a < network.weightMatrixCount; a++) {
            adjustments[a] = (float*)malloc(network.weights[a].rows * network.weights[a].cols * sizeof(float));
        }

        log_line("Initialised adjustment storage");

        return adjustments;
    }

    static void free_adjustments_storage(float** adjustmentsStorage, int weightMatrixCount) {
        log_line("Freeing adjustment storage...");

        for (int i = 0; i < weightMatrixCount; i++) {
            free(adjustmentsStorage[i]);
        }
        free(adjustmentsStorage);
        log_line("Freed adjustment storage");
    }

    static float calculate_error_derivative_for_final_layer(float* error_derivative_for_final_layer, matrix finalLayer, matrix expectedLayer) {

        float error = 0;
        for (int b = 0; b < finalLayer.rows; b++) {
            error_derivative_for_final_layer[b] = -(expectedLayer.values[b] - finalLayer.values[b]) * calculate_logistic_derivative(finalLayer.values[b]);
            error += 0.5 * (expectedLayer.values[b] - finalLayer.values[b]) * (expectedLayer.values[b] - finalLayer.values[b]);
        }
        log_line("Calculated derivatives for final layer.");

        return error;
    }

    static void update_weights(network network, float** adjustmentsStorage) {
        log_line("Updating weights...");
        for (int a = network.weightMatrixCount - 1; a >= 0; a--) {
            matrix weightMatrix = network.weights[a];

            for (int y = 0; y < weightMatrix.rows; y++) {
                for (int p = 0; p < weightMatrix.cols; p++) {
                    //log_line("row, col: %d, %d", y, p);
                    float* wij = &weightMatrix.values[weightMatrix.cols * y + p];
                    *wij = *wij - 0.01 * adjustmentsStorage[a][weightMatrix.cols * y + p];
                }
            }
        }
    }

    static void get_error_derivative_for_output_layer(matrix weightMatrix, matrix outputLayer, float* dError_dLayerAbove, float* dError_dOutputCurrent) {
        log_line("Calculating error derivative with respect to current output layer.");
        for (int j = 0; j < weightMatrix.cols; j++) {
            dError_dOutputCurrent[j] = 0;
            for (int i = 0; i < weightMatrix.rows; i++) {
                dError_dOutputCurrent[j] += dError_dLayerAbove[i] * calculate_logistic_derivative(outputLayer.values[i]) * weightMatrix.values[i * weightMatrix.cols + j];
            }
        }

        log_message("dError_dOutputCurrent: ");
        log_float_array(dError_dOutputCurrent, weightMatrix.cols);
    }

    static void update_error_derivative_for_layer_above(float* dError_dLayerAbove, float* dError_dOutputCurrent, int length) {
        if (dError_dLayerAbove != NULL) {
            free(dError_dLayerAbove);
        }

        dError_dLayerAbove = dError_dOutputCurrent;

        log_message("dError_dLayerAbove: ");
        log_float_array(dError_dLayerAbove, length);
        log_newline();
    }

    static void get_adjustments_for_layer(network network, int a, float* dError_dLayerAbove, float** adjustmentsStorage) {
        log_line("Calcuating loop for weight matrix: %d", a);

        matrix weightMatrix = network.weights[a];
        log_line("Weight Matrix: ");
        log_matrix(weightMatrix);

        matrix inputLayer = network.layers[a];
        log_line("Input Layer: ");
        log_matrix(inputLayer);

        matrix outputLayer = network.layers[a + 1];
        log_line("Output Layer: ");
        log_matrix(outputLayer);

        float* dError_dOutputCurrent = (float*)malloc(weightMatrix.cols * sizeof(float));
        get_error_derivative_for_output_layer(weightMatrix, outputLayer, dError_dLayerAbove, dError_dOutputCurrent);

        log_line("Calculating adjustments.");
        for (int i = 0; i < weightMatrix.rows; i++) {
            for (int j = 0; j < weightMatrix.cols; j++) {

                //log_line("row, col: %d, %d", i, j);
                log_number(i);
                log_number(j);
                float dOutputCurrentJ_dWeightIJ = inputLayer.values[j];
                log_message("Checkpoint...");
                float daij = dError_dOutputCurrent[j] * dOutputCurrentJ_dWeightIJ;
                log_message("Checkpoint...");
                adjustmentsStorage[a][weightMatrix.cols * i + j] = daij;
                log_message("Checkpoint...");
            }
        }

        update_error_derivative_for_layer_above(dError_dLayerAbove, dError_dOutputCurrent, weightMatrix.cols);
    }

    static void get_adjustments(network network, float* dError_dLayerAbove, float** adjustmentsStorage) {
        for (int a = network.weightMatrixCount - 1; a >= 0; a--) {
            get_adjustments_for_layer(network, a, dError_dLayerAbove, adjustmentsStorage);
        }
    }

    float train_network(network network, matrix expectedLayer) {

        float* dError_dLayerAbove = (float*)malloc(network.layers[network.layerCount - 1].rows * sizeof(float));
        float error = calculate_error_derivative_for_final_layer(dError_dLayerAbove, network.layers[network.layerCount - 1], expectedLayer);
        float** adjustmentsStorage = initialise_adjustments_storage(network);

        get_adjustments(network, dError_dLayerAbove, adjustmentsStorage);
        update_weights(network, adjustmentsStorage);
        free_adjustments_storage(adjustmentsStorage, network.weightMatrixCount);

        free(dError_dLayerAbove);

        return error;
    }
}