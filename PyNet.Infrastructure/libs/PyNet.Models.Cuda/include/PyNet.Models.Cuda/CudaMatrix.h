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
	
	static auto factory(std::shared_ptr<PyNet::DI::Context> context) {
		return new CudaMatrix{ context };
	}

	typedef PyNet::Models::Matrix base;

	CudaMatrix(std::shared_ptr<PyNet::DI::Context> context);
	std::unique_ptr<Matrix> operator*(const Matrix& m) const override;
	std::unique_ptr<Matrix> operator*(const double d) override;
	std::unique_ptr<Matrix> operator-(const Matrix& m) override;
	void operator+=(const Matrix& m) override;
};

