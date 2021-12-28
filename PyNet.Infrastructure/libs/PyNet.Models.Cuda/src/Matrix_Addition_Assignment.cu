#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_array.h"
#include <vector>
#include <stdlib.h>
#include <stddef.h>
#include "Matrix_Operations.h"

__global__ void matrixAdditionAssignmentKernel(double* A, double* B, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        A[i * cols + j] = A[i * cols + j] + B[i * cols + j];
    }
}


void internalMatrixAdditionAssignment(double* A, double* B, int rows, int cols) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(rows, cols);
    dim3 blocksPerGrid(1, 1);
    if (rows > 32) {
        threadsPerBlock.x = 32;
        blocksPerGrid.x = static_cast<int>(ceil(double(rows) / double(threadsPerBlock.x)));
    }

    if (cols > 32) {
        threadsPerBlock.y = 32;
        blocksPerGrid.y = static_cast<int>(ceil(double(cols) / double(threadsPerBlock.y)));
    }

    matrixAdditionAssignmentKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, rows, cols);
    cudaDeviceSynchronize();
}

void matrix_addition_assignment(Matrix& A, const Matrix& B) {

    cuda_array<double> d_A(A.GetCValues().size());
    cuda_array<double> d_B(B.GetCValues().size());

    d_A.set(A.GetValues());
    d_B.set(B.GetCValues());

    internalMatrixAdditionAssignment(d_A.getData(), d_B.getData(), A.GetRows(), A.GetCols());
    d_A.get(A.GetValues().data(), A.GetSize());
}
