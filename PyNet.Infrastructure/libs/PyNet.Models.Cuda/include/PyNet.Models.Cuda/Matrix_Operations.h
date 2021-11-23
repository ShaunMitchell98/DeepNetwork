#pragma once

#include <vector>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;

void cuda_matrix_multiply(const Matrix& A, const Matrix& B, Matrix& C);
void multiply_matrix_and_double(const Matrix& A, const double B, Matrix& C);
void matrix_subtract(const Matrix& A, const Matrix& B, Matrix& C);
void matrix_addition_assignment(Matrix& A, const Matrix& B);