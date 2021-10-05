#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_array.h"
#include <vector>
#include <stdlib.h>
#include <stddef.h>
#include "../Include/PyNet.Infrastructure.Cuda/Matrix_Multiplication.h"

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

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, Acols, Bcols);
    cudaDeviceSynchronize();
}

void cuda_matrix_multiply(std::vector<double> A, std::vector<double> B, std::vector<double> C, int Acols, int Bcols) {

    cuda_array<double> d_A(A.size());
    cuda_array<double> d_B(B.size());
    cuda_array<double> d_C(C.size());

    d_A.set(A);
    d_B.set(B);

    internalMatrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), Acols, Bcols);
    d_C.get(C.data(), C.size());
}
