#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../dev_array.h"
#include "../network.h"
#include "matrix_multiplication.h"
#include "../Activation Functions/logistic_function.h"
#include <stdlib.h>
#include <vector>
#include "../Logging/log.h"
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
    void matrix_multiply(matrix A, matrix B, matrix C) {

        dev_array<float> d_A(A.rows * A.cols);
        dev_array<float> d_B(B.rows * B.cols);
        dev_array<float> d_C(C.rows * C.cols);

        //log_matrix(A);
        //log_matrix(B);

        d_A.set(A.values, A.rows * A.cols);
        d_B.set(B.values, B.rows * B.cols);  

        internalMatrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), A.cols, B.cols);
        d_C.get(C.values, C.rows * C.cols);
    }
}