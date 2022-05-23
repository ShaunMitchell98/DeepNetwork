#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdlib.h>
#include <stddef.h>
#include "CudaArray.h"
#include "Matrix_Operations.h"

using namespace std;

__global__ void matrixExpKernel(double* A, double* B, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        B[i * cols + j] = exp(A[i * cols + j]);
    }
}

void internalMatrixExp(double* A, double* B, int rows, int cols) {

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

    matrixExpKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, rows, cols);
    cudaDeviceSynchronize();
}

void matrix_exp(const vector<double> A, vector<double>& B, int Arows, int Acols) {

    CudaArray<double> d_A(A.size());
    CudaArray<double> d_B(B.size());

    d_A.set(A);
    d_B.set(B);

    internalMatrixExp(d_A.getData(), d_B.getData(), Arows, Acols);
    d_B.get(B.data(), B.size());
}
