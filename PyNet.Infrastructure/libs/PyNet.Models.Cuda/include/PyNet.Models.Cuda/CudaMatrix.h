#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Context.h"

using namespace PyNet::Models;

class __declspec(dllexport) CudaMatrix
#ifdef CUDA_VECTOR
	: public virtual PyNet::Models::Matrix
#else
	: public PyNet::Models::Matrix
#endif
{
public:
	
	static auto factory(di::Context& context) {
		return new CudaMatrix{ context };
	}

	typedef PyNet::Models::Matrix base;

	CudaMatrix(di::Context& context);
	Matrix& operator*(const Matrix& m) const override;
	Matrix& operator*(const double d) override;
	Matrix& operator-(const Matrix& m) override;
	void operator+=(const Matrix& m) override;
};

