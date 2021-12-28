#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_array.h"
#include "Matrix_Operations.h"

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int Arows, int Acols, int Bcols) {
    int ROW = blockIdx.x * blockDim.x + threadIdx.x;
    int COL = blockIdx.y * blockDim.y + threadIdx.y;

    double tmpSum = 0;

    if (ROW < Arows && COL < Bcols) {
        for (auto i = 0; i < Acols; i++) {
            tmpSum += A[ROW * Acols + i] * B[i * Bcols + COL];
        }

        C[ROW * Bcols + COL] = tmpSum;
    }
}

void internalMatrixMultiply(double* A, double* B, double* C, int Arows, int Acols, int Bcols) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(Arows, Bcols);
    dim3 blocksPerGrid(1, 1);
    if (Arows > 32) {
        threadsPerBlock.x = 32;
        blocksPerGrid.x = static_cast<int>(ceil(double(Arows) / double(threadsPerBlock.x)));
    }

    if (Bcols > 32) {
        threadsPerBlock.y = 32;
        blocksPerGrid.y = static_cast<int>(ceil(double(Bcols) / double(threadsPerBlock.y)));
    }

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, Arows, Acols, Bcols);
    cudaDeviceSynchronize();
}

void cuda_matrix_multiply(const Matrix& A, const Matrix& B, Matrix& C) {

    cuda_array<double> d_A(A.GetCValues().size());
    cuda_array<double> d_B(B.GetCValues().size());
    cuda_array<double> d_C(C.GetCValues().size());

    d_A.set(A.GetCValues());
    d_B.set(B.GetCValues());

    internalMatrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), A.GetRows(), A.GetCols(), B.GetCols());
    d_C.get(C.GetValues().data(), C.GetSize());
}