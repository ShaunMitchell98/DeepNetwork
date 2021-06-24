#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrix_multiplication.h"
#include "dev_array.h"
#include <stdlib.h>
#include <vector>
#include <stdio.h>

using namespace std;

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}

extern "C"
{
    void internalMatrixMultiply(float* A, float* B, float* C, int N) {

        // declare the number of blocks per grid and the number of threads per block
        // use 1 to 512 threads per block
        dim3 threadsPerBlock(N, N);
        dim3 blocksPerGrid(1, 1);
        if (N * N > 512) {
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
        }

        matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        cudaDeviceSynchronize();
    }

    void matrixMultiply(float* A, float* B, float* C, int N) {

        int size = N * N;

        dev_array<float> d_A(size);
        dev_array<float> d_B(size);
        dev_array<float> d_C(size);

        d_A.set(A, size);
        d_B.set(B, size);

        internalMatrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), N);
        d_C.get(C, size);
    }
}