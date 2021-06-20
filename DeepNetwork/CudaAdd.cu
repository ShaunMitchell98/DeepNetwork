#include "CudaAdd.h"

#include <stdio.h>


__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

CudaAdder::CudaAdder() {
    dev_a = 0;
    dev_b = 0; 
    dev_c = 0;
}

cudaError_t CudaAdder::handleError() {
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cudaStatus;
}

cudaError_t CudaAdder::allocateGPUMemory(int** dev, int size) {
    cudaStatus = cudaMalloc((void**) dev, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return handleError();
    }
}

cudaError_t CudaAdder::copyVectorToGPU(int* dev, const int* input, int size) {
    cudaStatus = cudaMemcpy(dev, input, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return handleError();
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t CudaAdder::addWithCuda(int* output, const int* input1, const int* input2, unsigned int size)
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return handleError();
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return handleError();
    }

    allocateGPUMemory(&dev_c, size);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    allocateGPUMemory(&dev_a, size);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    allocateGPUMemory(&dev_b, size);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers.
    copyVectorToGPU(dev_a, input1, size);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    copyVectorToGPU(dev_b, input2, size);

    if (cudaStatus != cudaSuccess) {
        return cudaStatus;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return handleError();
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return handleError();
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return handleError();
    }

    return cudaStatus;
}