#pragma once

#define CUDA_VECTOR

#include "PyNet.Models/Vector.h"
#include "CudaMatrix.h"

class __declspec(dllexport) CudaVector : public Vector, private CudaMatrix {
public:

	static auto factory(di::Context& context, PyNet::Models::Activation& activation) {
		return new CudaVector{ context, activation };
	}

	typedef Vector base;

	CudaVector(di::Context& context, PyNet::Models::Activation& activation);

	Matrix& operator*(const Matrix& m) const override {
		return CudaMatrix::operator*(m);
	}

	Matrix& operator-(const Matrix& m) override {
		return CudaMatrix::operator-(m);
	}

	void operator+=(const Matrix& m) override {
		CudaMatrix::operator+=(m);
	}

	void operator+=(const Vector& v) override {
		return CudaMatrix::operator+=((Matrix&)v);
	}

	Vector& operator*(const double d) override;

	Vector& operator-(const Vector& v) override;

	int GetRows() const override {
		return Vector::GetRows();
	}

	int GetCols() const override {
		return Vector::GetCols();
	}

	operator Matrix& () {
		return static_cast<PyNet::Models::Matrix&>(static_cast<Vector&>(*this));
	}

	CudaVector(const CudaVector& v);
};

#undef CUDA_VECTOR