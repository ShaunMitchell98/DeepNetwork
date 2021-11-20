#pragma once

#include "PyNet.Models/Vector.h"
#include "CpuMatrix.h"

class CpuVector : public PyNet::Models::Vector, public CpuMatrix {

public:
	static auto factory(di::Context& context, Activation& activation) {
		return CpuVector{ context, activation };
	}

	typedef PyNet::Models::Vector base;

	CpuVector(di::Context& context, Activation& activation) : Vector(context, activation), CpuMatrix(context) {}

	Matrix& operator*(const Matrix& m) {
		return CpuMatrix::operator*(m);
	}

	Matrix& operator-(const Matrix& m) {
		return CpuMatrix::operator-(m);
	}

	void operator+=(const Matrix& m) {
		CpuMatrix::operator+=(m);
	}

	void operator+=(const Vector& v) {
		return CpuMatrix::operator+=(v);
	}

	Vector& operator*(const double d) {
		return static_cast<Vector&>(CpuMatrix::operator*(d));
	}

	Vector& operator-(const Vector& v) override {
		return static_cast<Vector&>(CpuMatrix::operator-(v));
	}
};