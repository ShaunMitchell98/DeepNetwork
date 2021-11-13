#pragma once

#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;

class CudaMatrix : public PyNet::Models::Matrix
{
public:
	
	static auto factory(di::Context& context) {
		return new CudaMatrix{ context };
	}

	CudaMatrix(di::Context& context) : Matrix(context) {}
	Matrix& operator*(const Matrix& m);
	Matrix& operator*(const double d);
	Matrix& operator-(const Matrix& m);
	void operator+=(const Matrix& m);
};

