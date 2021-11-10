#pragma once

#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;

class CpuMatrix : public Matrix
{
public:
	CpuMatrix(int rows, int cols) : Matrix(rows, cols) {}
	Matrix& operator*(const Matrix& m);
	Matrix& operator*(const double d);
	Matrix& operator-(const Matrix& m);
	void operator+=(const Matrix& m);
};