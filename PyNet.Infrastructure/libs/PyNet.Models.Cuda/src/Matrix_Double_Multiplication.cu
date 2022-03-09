#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <vector>
#include "CudaArray.h"
#include "Matrix_Operations.h"

using namespace std;

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

    if (Arows > 32) {
        threadsPerBlock.x = 32;
        blocksPerGrid.x = static_cast<int>(ceil(double(Arows) / double(threadsPerBlock.x)));
    }

    if (Acols > 32) {
        threadsPerBlock.y = 32;
        blocksPerGrid.y = static_cast<int>(ceil(double(Acols) / double(threadsPerBlock.y)));
    }

    matrixDoubleMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, Arows, Acols);
    cudaDeviceSynchronize();
}

void multiply_matrix_and_double(const vector<double>& A, const double B, vector<double>& C, int Arows, int Acols) {

    const std::vector<double> bVector{ B };

    CudaArray<double> d_A(A.size());
    CudaArray<double> d_B(bVector.size());
    CudaArray<double> d_C(C.size());

    d_A.set(A);
    d_B.set(bVector);

    internalMatrixDoubleMultiply(d_A.getData(), d_B.getData(), d_C.getData(), Arows, Acols);
    d_C.get(C.data(), C.size());
}