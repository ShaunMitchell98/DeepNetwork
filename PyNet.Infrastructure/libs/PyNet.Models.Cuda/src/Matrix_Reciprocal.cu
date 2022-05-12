#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <vector>
#include "CudaArray.h"
#include "Matrix_Operations.h"

using namespace std;

__global__ static void matrixReciprocalKernel(double* A, double* B, int Arows, int Acols) {
    int ROW = blockIdx.x * blockDim.x + threadIdx.x;
    int COL = blockIdx.y * blockDim.y + threadIdx.y;

    if (ROW < Arows && COL < Acols) {
        B[ROW * Acols + COL] = 1 / A[ROW * Acols + COL];
    }
}

static void internalMatrixReciprocal(double* A, double* B, int Arows, int Acols) {

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

    matrixReciprocalKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, Arows, Acols);
    cudaDeviceSynchronize();
}

void matrix_reciprocal(const vector<double>& A, vector<double>& B, int Arows, int Acols) {

    CudaArray<double> d_A(A.size());
    CudaArray<double> d_B(B.size());

    d_A.set(A);

    internalMatrixReciprocal(d_A.getData(), d_B.getData(), Arows, Acols);
    d_B.get(B.data(), B.size());
}