#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../dev_array.h"
#include "matrix_multiplication.h"
#include <stdlib.h>
#include <stddef.h>

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int Acols, int Bcols) {
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    double tmpSum = 0;

    if (ROW < Acols && COL < Bcols) {
        // each thread computes one element of the block sub-matrix
        for (auto i = 0; i < Acols; i++) {
            tmpSum += A[ROW * Acols + i] * B[i * Bcols + COL];
        }

        C[ROW * Bcols + COL] = tmpSum;
    }
}

void internalMatrixMultiply(double* A, double* B, double* C, int Acols, int Bcols) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(Acols, Acols);
    dim3 blocksPerGrid(1, 1);
    if (Acols * Acols > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = static_cast<int>(ceil(double(Acols) / double(threadsPerBlock.x)));
        blocksPerGrid.y = static_cast<int>(ceil(double(Acols) / double(threadsPerBlock.y)));
    }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, Acols, Bcols);
    cudaDeviceSynchronize();
}

void matrix_multiply(Matrix* A, Matrix* B, Matrix* C) {

    for (auto i = 0; i < A->Rows; i++) {
        for (auto j = 0; j < B->Cols; j++) {
            double tempValue = 0;
            for (auto k = 0; k < A->Cols; k++) {
                tempValue += A->GetValue(i, k) * B->GetValue(k, j);
            }

            C->SetValue(j, i, tempValue);
        }
    }

    //dev_array<double> d_A(A->Rows * A->Cols);
    //dev_array<double> d_B(B->Rows * B->Cols);
    //dev_array<double> d_C(C->Rows * C->Cols);

    //d_A.set(A->Values.data(), static_cast<size_t>(A->Rows * A->Cols));
    //d_B.set(B->Values.data(), static_cast<size_t>(B->Rows * B->Cols));  

    //internalMatrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), A->Cols, B->Cols);
    //d_C.get(C->Values.data(), static_cast<size_t>(C->Rows * C->Cols));
}