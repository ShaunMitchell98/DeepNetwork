#pragma once

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class dev_array {
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
			cudaFree(start_);
			start_ = end_ = 0;
		}
	}

public:

	explicit dev_array()
		: start_(0),
		end_(0)
	{}

	explicit dev_array(size_t size) {
		allocate(size);
	}

	~dev_array() {
		free();
	}

	T* getData() {
		return start_;
	}

	void resize(size_t size) {
		free();
		allocate(size);
	}

	void set(const T* src, size_t size) {
		size_t min = std::min(size, getSize());
		cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);

		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to copy to device memory.");
		}
	}

	void get(T* dest, size_t size) {
		size_t min = std::min(size, getSize());

		cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);

		if (result != cudaSuccess) {
			throw std::runtime_error("Failed to copy to host memory");
		}
	}
};