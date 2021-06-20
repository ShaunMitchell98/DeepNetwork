#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaAdder {
private:
	int* dev_a;
	int* dev_b;
	int* dev_c;
	cudaError_t cudaStatus;
	cudaError_t handleError();
	cudaError_t allocateGPUMemory(int** dev, int size);
	cudaError_t copyVectorToGPU(int* dev, const int* input, int size);

public:
	CudaAdder();
	cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
};
