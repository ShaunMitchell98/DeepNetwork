#pragma once


#include "PyNet.Models/Vector.h"
#include "CudaMatrix.h"

class CudaVector : public Vector, public CudaMatrix {
public:

	static auto factory(di::Context& context, PyNet::Models::Activation& activation) {
		return new CudaVector{ context, activation };
	}

	typedef Vector base;

	CudaVector(di::Context& context, PyNet::Models::Activation& activation) : Vector(context, activation), CudaMatrix(context) {}

	Matrix& operator*(const Matrix& m) override {
		return CudaMatrix::operator*(m);
	}

	Matrix& operator-(const Matrix& m) override {
		return CudaMatrix::operator-(m);
	}

	void operator+=(const Matrix& m) override {
		CudaMatrix::operator+=(m);
	}

	void operator+=(const Vector& v) override {
		return CudaMatrix::operator+=(v);
	}

	Vector& operator*(const double d) override {
		return static_cast<Vector&>(CudaMatrix::operator*(d));
	}

	Vector& operator-(const Vector& v) override {
		return static_cast<Vector&>(CudaMatrix::operator-(v));
	}
};