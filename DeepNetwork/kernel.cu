#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "matrix_multiplication.h"
#include "dev_array.h"
#include <math.h>
int main()
{
    int N = 2;
    int size = N * N;
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    std::vector<float> h_C(size);

    h_A[0] = 1;
    h_A[1] = 1;
    h_A[2] = 1;
    h_A[3] = 1;


    h_B[0] = 1;
    h_B[1] = 1;
    h_B[2] = 1;
    h_B[3] = 1;



    dev_array<float> d_A(size);
    dev_array<float> d_B(size);
    dev_array<float> d_C(size);

    d_A.set(&h_A[0], size);
    d_B.set(&h_B[0], size);

    matrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], size);
    cudaDeviceSynchronize();

    return 0;
}


