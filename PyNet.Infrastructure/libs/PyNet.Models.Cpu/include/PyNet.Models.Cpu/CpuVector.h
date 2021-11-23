#pragma once

#define CPU_VECTOR

#include "PyNet.Models/Vector.h"
#include "CpuMatrix.h"

class CpuVector : public PyNet::Models::Vector, public CpuMatrix {

public:
	static auto factory(di::Context& context, Activation& activation) {
		return new CpuVector{ context, activation };
	}

	typedef PyNet::Models::Vector base;

	CpuVector(di::Context& context, Activation& activation);

	Matrix& operator*(const Matrix& m) const  {
		return CpuMatrix::operator*(m);
	}

	Matrix& operator-(const Matrix& m) {
		return CpuMatrix::operator-(m);
	}

	void operator+=(const Matrix& m) {
		CpuMatrix::operator+=(m);
	}

	void operator+=(const Vector& v) override {
		return CpuMatrix::operator+=(v);
	}

	Vector& operator*(const double d) override {
		return dynamic_cast<CpuVector&>(CpuMatrix::operator*(d));
	}

	Vector& operator-(const Vector& v) override {
		return dynamic_cast<CpuVector&>(CpuMatrix::operator-(v));
	}

	int GetRows() const override {
		return Vector::GetRows();
	}

	int GetCols() const override {
		return Vector::GetCols();
	}

	explicit operator Matrix& () {
		return static_cast<PyNet::Models::Matrix&>(static_cast<Vector&>(*this));
	}
};

#undef CPU_VECTOR