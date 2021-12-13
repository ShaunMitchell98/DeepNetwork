#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"

using namespace PyNet::Models;

class CpuMatrix 
#ifdef CPU_VECTOR
	: public virtual PyNet::Models::Matrix
#else
	: public PyNet::Models::Matrix
#endif
{
public:

	static auto factory(std::shared_ptr<PyNet::DI::Context> context) {
		return new CpuMatrix{ context };
	}

	typedef PyNet::Models::Matrix base;

	CpuMatrix(std::shared_ptr<PyNet::DI::Context> context);
	std::unique_ptr<Matrix> operator*(const Matrix& m) const override;
	std::unique_ptr<Matrix> operator*(const double d) override;
	std::unique_ptr<Matrix> operator-(const Matrix& m) override;
	void operator+=(const Matrix& m) override;
};