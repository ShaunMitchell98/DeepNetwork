#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"

using namespace PyNet::Models;

class __declspec(dllexport) CudaMatrix
#ifdef CUDA_VECTOR
	: public virtual PyNet::Models::Matrix
#else
	: public PyNet::Models::Matrix
#endif
{
public:
	
	static auto factory() {
		return new CudaMatrix();
	}

	typedef PyNet::Models::Matrix base;

	CudaMatrix();
	std::unique_ptr<Matrix> operator*(const Matrix& m) const override;
	std::unique_ptr<Matrix> operator*(const double d) override;
	std::unique_ptr<Matrix> operator-(const Matrix& m) override;
	std::unique_ptr<Matrix> operator~() override;
	void operator+=(const Matrix& m) override;
};

