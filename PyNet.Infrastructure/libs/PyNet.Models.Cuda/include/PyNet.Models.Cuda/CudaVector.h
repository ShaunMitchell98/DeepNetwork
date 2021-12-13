#pragma once

#define CUDA_VECTOR

#include "PyNet.Models/Vector.h"
#include "CudaMatrix.h"

class __declspec(dllexport) CudaVector : public Vector, private CudaMatrix {
public:

	static auto factory(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation) {
		return new CudaVector{ context, activation };
	}

	typedef Vector base;

	CudaVector(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation);

	std::unique_ptr<Matrix> operator*(const Matrix& m) const override {
		return CudaMatrix::operator*(m);
	}

	std::unique_ptr<Matrix> operator-(const Matrix& m) override {
		return CudaMatrix::operator-(m);
	}

	void operator+=(const Matrix& m) override {
		CudaMatrix::operator+=(m);
	}

	void operator+=(const Vector& v) override {
		return CudaMatrix::operator+=((Matrix&)v);
	}

	std::unique_ptr<Matrix> operator*(const double d) override {
		return CudaMatrix::operator*(d);
	}

	std::unique_ptr<Vector> operator-(const Vector& v) override {
		return std::unique_ptr<Vector>(dynamic_cast<Vector*>(CudaMatrix::operator-(v).get()));
	}

	int GetRows() const override {
		return Vector::GetRows();
	}

	int GetCols() const override {
		return Vector::GetCols();
	}

	CudaVector(const CudaVector& v);
};

#undef CUDA_VECTOR