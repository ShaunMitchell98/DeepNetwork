#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dev_array.h"
#include "network.h"
#include "matrix_multiplication.h"
#include <stdlib.h>
#include <vector>
#include "Log.h"
#include <stdio.h>

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int Acols, int Bcols) {
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < Acols && COL < Bcols) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < Acols; i++) {
            tmpSum += A[ROW * Acols + i] * B[i * Bcols + COL];
        }

        C[ROW * Bcols + COL] = tmpSum;
    }
}

void internalMatrixMultiply(float* A, float* B, float* C, int Acols, int Bcols) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(Acols, Acols);
    dim3 blocksPerGrid(1, 1);
    if (Acols * Acols > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(Acols) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(Acols) / double(threadsPerBlock.y));
    }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, Acols, Bcols);
    cudaDeviceSynchronize();
}

extern "C"
{
    void Log(char* buf, float thing, FILE* fp) {
        gcvt(thing, 5, buf);
        fwrite(buf, sizeof(char), 5, fp);
        fwrite("\n", sizeof(char), 1, fp);
    }

    void matrix_multiply(matrix A, matrix B, matrix C) {

        dev_array<float> d_A(A.rows * A.cols);
        dev_array<float> d_B(B.rows * B.cols);
        dev_array<float> d_C(C.rows * C.cols);

        //LogMatrix(A);
        //LogMatrix(B);

        d_A.set(A.values, A.rows * A.cols);
        d_B.set(B.values, B.rows * B.cols);  

        internalMatrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), A.cols, B.cols);
        d_C.get(C.values, C.rows * C.cols);
    }

    ////Error function = 0.5* Sum((error - actual) ^2)
    ////This works for the final layer
    void train_layer(matrix weights, matrix inputLayer, matrix outputLayer, matrix expectedLayer) {

        LogMatrix(weights);
        for (int i = 0; i < outputLayer.rows; i++) {
            for (int j = 0; j < inputLayer.rows; j++) {
                int dij = (expectedLayer.values[i] - outputLayer.values[i]) * inputLayer.values[j];
                float* wij = &weights.values[weights.cols * i + j];
                *wij = *wij - 0.01 * dij;
            }
        }

        LogMatrix(weights);
    }

    float Error(matrix expectedOutput, matrix actualOutput) {
        
        float sum = 0;
        for (int i = 0; i < expectedOutput.rows; i++) {
            sum += 0.5 * (expectedOutput.values[i] - actualOutput.values[i]) * (expectedOutput.values[i] - actualOutput.values[i]);
        }

        return sum;
    }

    int GetNumberOfWeights(network network) {
        int numberOfWeights = 0;

        for (int a = 0; a < network.weightMatrixCount; a++) {
            matrix current_matrix = network.weights[a];
            numberOfWeights += current_matrix.rows * current_matrix.cols;
        }

        LogMessage("Number of weights: ");
        LogNumber(numberOfWeights);
        LogNewline();

        return numberOfWeights;

    }

    float** InitialiseAdjustmentsStorage(network network) {
        LogLine("Initialising adjustment storage...");

        float** adjustments = (float**)malloc(network.weightMatrixCount * sizeof(float*));

        for (int a = 0; a < network.weightMatrixCount; a++) {
            adjustments[a] = (float*)malloc(network.weights[a].rows * network.weights[a].cols * sizeof(float));
        }

        LogLine("Initialised adjustment storage");

        return adjustments;
    }

    void FreeAdjustmentsStorage(float** adjustmentsStorage, int weightMatrixCount) {
        LogLine("Freeing adjustment storage...");

        for (int i = 0; i < weightMatrixCount; i++) {
            free(adjustmentsStorage[i]);
        }
        free(adjustmentsStorage);
        LogLine("Freed adjustment storage");
    }

    float train_network(network network, matrix expectedLayer) {

        DeleteLogFile();
        int numberOfNeuronsInFinalLayer = network.layers[network.layerCount - 1].rows;
        LogMessage("Number of neurons in final layer: ");
        LogNumber(numberOfNeuronsInFinalLayer);
        LogNewline();

        float* dError_dLast = (float*)malloc(numberOfNeuronsInFinalLayer * sizeof(float));

        int numberOfWeights = GetNumberOfWeights(network);

        float** adjustmentsStorage = InitialiseAdjustmentsStorage(network);

        float error = 0;
        matrix finalLayer = network.layers[network.layerCount - 1];
        for (int b = 0; b < numberOfNeuronsInFinalLayer; b++) {
            dError_dLast[b] = -(expectedLayer.values[b] - finalLayer.values[b]);
            error += 0.5 * (expectedLayer.values[b] - finalLayer.values[b]) * (expectedLayer.values[b] - finalLayer.values[b]);
        }

        float* dError_dLayerAbove = dError_dLast;

        LogLine("Calculated derivatives for final layer.");

        for (int a = network.weightMatrixCount - 1; a >= 0; a--) {

            LogMessage("Calcuating loop for weight matrix: ");
            LogNumber(a);
            LogNewline();

            matrix weightMatrix = network.weights[a];

            LogLine("Weight Matrix: ");
            LogMatrix(weightMatrix);


            matrix inputLayer = network.layers[a];

            LogLine("Input Layer: ");
            LogMatrix(inputLayer);

            matrix outputLayer = network.layers[a + 1];

            LogLine("Output Layer: ");
            LogMatrix(outputLayer);

            float* dError_dOutputCurrent = (float*)malloc(weightMatrix.cols * sizeof(float));

            LogLine("Calculating error derivative with respect to current output layer.");
            for (int j = 0; j < weightMatrix.cols; j++) {
                dError_dOutputCurrent[j] = 0;
                for (int i = 0; i < weightMatrix.rows; i++) {
                    dError_dOutputCurrent[j] += dError_dLayerAbove[i] * weightMatrix.values[i * weightMatrix.cols + j];
                }
            }

            LogMessage("dError_dOutputCurrent: ");
            LogFloatArray(dError_dOutputCurrent, weightMatrix.cols);

            LogLine("Calculating adjustments.");
            for (int i = 0; i < weightMatrix.rows; i++) {
                for (int j = 0; j < weightMatrix.cols; j++) {

                    LogMessage("i, j: ");
                    LogNumber(i);
                    LogMessage(", ");
                    LogNumber(j);
                    LogNewline();

                    float dOutputJ_dWeightIJ = inputLayer.values[j];
                    float daij = dError_dOutputCurrent[j] * dOutputJ_dWeightIJ;
                    adjustmentsStorage[a][weightMatrix.cols * i + j] = daij;
                }
            }


            if (dError_dLayerAbove != NULL) {
                LogLine("Freeing error derivative with respect to layer above");
                free(dError_dLayerAbove);
                LogLine("Freed error derivative with respect to layer above.");
            }

            dError_dLayerAbove = dError_dOutputCurrent;

            LogMessage("dError_dLayerAbove: ");
            LogFloatArray(dError_dLayerAbove, weightMatrix.cols);
            LogNewline();
        }

        LogLine("Updating weights...");
        for (int a = network.weightMatrixCount - 1; a >= 0; a--) {
            matrix weightMatrix = network.weights[a];
  
            for (int y = 0; y < weightMatrix.rows; y++) {
                for (int p = 0; p < weightMatrix.cols; p++) {
                    LogMessage("row, col: ");
                    LogNumber(y);
                    LogMessage(", ");
                    LogNumber(p);
                    LogNewline();
                    float* wij = &weightMatrix.values[weightMatrix.cols * y + p];          
                    *wij = *wij - 0.01 * adjustmentsStorage[a][weightMatrix.cols * y + p];
                }
            } 
        }
    

        FreeAdjustmentsStorage(adjustmentsStorage, network.weightMatrixCount);
        
        return error;
    }

}