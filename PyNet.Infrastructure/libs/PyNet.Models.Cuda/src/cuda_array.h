#pragma once

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>

template <class T>
class cuda_array {
private:
	T* start_;
	T* end_;

	size_t getSize() {
		return end_ - start_;
	}

	void allocate(size_t size) {
		cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));

		if (result != cudaSuccess) {
			start_ = end_ = 0;
			throw std::runtime_error("Failed to allocate device memory");
		}

		end_ = start_ + size;
	}

	void free() {
		if (start_ != 0) {
			auto error = cudaFree(start_);


			if (error != cudaSuccess) {
				start_ = end_ = 0;
				throw std::runtime_error("Failed to allocate device memory");
			}

			start_ = end_ = 0;
		}
	}

public:

	explicit cuda_array()
		: start_(0),
		end_(0)
	{}

	explicit cuda_array(size_t size) {
		allocate(size);
	}

	~cuda_array() {
		free();
	}

	T* getData() {
		return start_;
	}

	void resize(size_t size) {
		free();
		allocate(size);
	}

	void set(const std::vector<T> vec) {
		size_t min = std::min(vec.size(), getSize());
		cudaError_t result = cudaMemcpy(start_, vec.data(), min * sizeof(T), cudaMemcpyHostToDevice);

		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to copy to device memory.");
		}
	}

	void get(T* dest, size_t size) {
		size_t min = std::min(size, getSize());

		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to copy to host memory");
		}
	}
};