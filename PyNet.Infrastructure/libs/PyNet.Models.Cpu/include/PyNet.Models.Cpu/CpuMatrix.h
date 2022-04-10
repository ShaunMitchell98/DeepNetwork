#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"

using namespace PyNet::Models;
using namespace std;

class CpuMatrix 
#ifdef CPU_VECTOR
	: public virtual Matrix
#else
	: public Matrix
#endif
{
public:

	static auto factory() {
		return new CpuMatrix();
	}

	typedef Matrix base;

	CpuMatrix();
	const double& operator()(size_t row, size_t col) const { return Matrix::operator()(row, col); }
	double& operator()(size_t row, size_t col) { return Matrix::operator()(row, col); }
	unique_ptr<Matrix> operator*(const Matrix& m) const override;
	unique_ptr<Matrix> operator*(const double d) const override;
	unique_ptr<Matrix> operator+(const Matrix& m) const override;
	unique_ptr<Matrix> operator-(const Matrix& m) const override;
	unique_ptr<Matrix> operator~() const override;
	void operator+=(const Matrix& m) override;
};