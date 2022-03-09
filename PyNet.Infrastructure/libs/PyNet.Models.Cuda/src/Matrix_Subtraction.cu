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

    if (rows > 32) {
        threadsPerBlock.x = 32;
        blocksPerGrid.x = static_cast<int>(ceil(double(rows) / double(threadsPerBlock.x)));
    }

    if (cols > 32) {
        threadsPerBlock.y = 32;
        blocksPerGrid.y = static_cast<int>(ceil(double(cols) / double(threadsPerBlock.y)));
    }

    matrixSubtractionKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, rows, cols);
    cudaDeviceSynchronize();
}

void matrix_subtract(const vector<double>& A, const vector<double>& B, vector<double>& C, int Arows, int Acols) {

    CudaArray<double> d_A(A.size());
    CudaArray<double> d_B(B.size());
    CudaArray<double> d_C(C.size());

    d_A.set(A);
    d_B.set(B);

    internalMatrixSubtract(d_A.getData(), d_B.getData(), d_C.getData(), Arows, Acols);
    d_C.get(C.data(), C.size());
}
