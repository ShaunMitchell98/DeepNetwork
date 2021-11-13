#pragma once

#include <vector>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;

extern "C" {

	void cuda_matrix_multiply(std::vector<double> A, std::vector<double>  B, std::vector<double> C, int Acols, int Bcols);
	void multiply_matrix_and_double(std::vector<double> A, double B, std::vector<double> C, int Acols, int Arows);
	void matrix_subtract(const Matrix& A, const Matrix& B, Matrix& C);
	void matrix_addition_assignment(Matrix& A, const Matrix& B);
}