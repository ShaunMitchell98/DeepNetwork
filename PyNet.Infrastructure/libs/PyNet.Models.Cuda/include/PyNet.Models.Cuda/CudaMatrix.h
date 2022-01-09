#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"

using namespace PyNet::Models;
using namespace std;

class __declspec(dllexport) CudaMatrix
#ifdef CUDA_VECTOR
	: public virtual Matrix
#else
	: public Matrix
#endif
{
public:
	
	static auto factory() {
		return new CudaMatrix();
	}

	typedef Matrix base;

	CudaMatrix();
	unique_ptr<Matrix> operator*(const Matrix& m) const override;
	unique_ptr<Matrix> operator*(const double d) const override;
	unique_ptr<Matrix> operator+(const Matrix& m) const override;
	unique_ptr<Matrix> operator-(const Matrix& m) const override;
	unique_ptr<Matrix> operator~() const override;
	void operator+=(const Matrix& m) override;
};