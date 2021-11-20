#pragma once

#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;

class __declspec(dllexport) CudaMatrix : public PyNet::Models::Matrix
{
public:
	
	static auto factory(di::Context& context) {
		return new CudaMatrix{ context };
	}

	typedef PyNet::Models::Matrix base;

	CudaMatrix(di::Context& context) : Matrix(context) {}
	Matrix& operator*(const Matrix& m) override;
	Matrix& operator*(const double d) override;
	Matrix& operator-(const Matrix& m) override;
	void operator+=(const Matrix& m) override;
};

