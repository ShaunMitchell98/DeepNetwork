#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_array.h"
#include <vector>
#include <stdlib.h>
#include <stddef.h>
#include "Matrix_Operations.h"

__global__ void matrixSubtractionKernel(double* A, double* B, double* C, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        C[i * cols + j] = A[i * cols + j] - B[i * cols + j];
    }
}


void internalMatrixSubtract(double* A, double* B, double* C, int rows, int cols) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(rows, cols);
    dim3 blocksPerGrid(1, 1);
    if (rows * cols > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = static_cast<int>(ceil(double(rows) / double(threadsPerBlock.x)));
        blocksPerGrid.y = static_cast<int>(ceil(double(cols) / double(threadsPerBlock.y)));
    }

    matrixSubtractionKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}

void matrix_subtract(const Matrix& A, const Matrix& B, Matrix& C) {

    cuda_array<double> d_A(A.GetCValues().size());
    cuda_array<double> d_B(B.GetCValues().size());
    cuda_array<double> d_C(C.GetCValues().size());

    d_A.set(A.GetCValues());
    d_B.set(B.GetCValues());

    internalMatrixSubtract(d_A.getData(), d_B.getData(), d_C.getData(), A.GetRows(), A.GetCols());
    d_C.get(C.GetValues().data(), C.GetSize());
}
