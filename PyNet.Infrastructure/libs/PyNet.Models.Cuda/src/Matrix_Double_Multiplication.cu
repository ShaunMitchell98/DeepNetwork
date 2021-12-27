#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_array.h"
#include "Matrix_Operations.h"

__global__ static void matrixDoubleMultiplicationKernel(double* A, double* B, double* C, int Arows, int Acols) {
    int ROW = blockIdx.x * blockDim.x + threadIdx.x;
    int COL = blockIdx.y * blockDim.y + threadIdx.y;

    if (ROW < Arows && COL < Acols) {
        C[ROW * Acols + COL] = A[ROW * Acols + COL] * *B;
    }
}

static void internalMatrixDoubleMultiply(double* A, double* B, double* C, int Arows, int Acols) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    dim3 threadsPerBlock(Arows, Acols);
    dim3 blocksPerGrid(1, 1);
    if (Arows * Acols > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = static_cast<int>(ceil(double(Arows) / double(threadsPerBlock.x)));
        blocksPerGrid.y = static_cast<int>(ceil(double(Acols) / double(threadsPerBlock.y)));
    }

    matrixDoubleMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, Arows, Acols);
    cudaDeviceSynchronize();
}

void multiply_matrix_and_double(const Matrix& A, const double B, Matrix& C) {

    const std::vector<double> bVector{ B };

    cuda_array<double> d_A(A.GetCValues().size());
    cuda_array<double> d_B(bVector.size());
    cuda_array<double> d_C(C.GetCValues().size());

    d_A.set(A.GetCValues());
    d_B.set(bVector);

    internalMatrixDoubleMultiply(d_A.getData(), d_B.getData(), d_C.getData(), A.GetRows(), A.GetCols());
    d_C.get(C.GetValues().data(), C.GetSize());
}