#include "CudaMatrix.h"
#include "Matrix_Operations.h"

CudaMatrix::CudaMatrix(di::Context& context)
#ifndef CUDA_VECTOR
	: Matrix(context)
#endif
{}

Matrix& CudaMatrix::operator*(const Matrix& m) const {
	auto& c = Context.get<Matrix>();
	cuda_matrix_multiply(*this, m, c);

	return c;
}

Matrix& CudaMatrix::operator*(const double d) {

	auto& c = Context.get<Matrix>();
	multiply_matrix_and_double(*this, d, c);
	return c;
}

Matrix& CudaMatrix::operator-(const Matrix& m) {
	auto& c = Context.get<Matrix>();
	matrix_subtract(*this, m, c);
	return c;
}

void CudaMatrix::operator+=(const Matrix& m) {

	matrix_addition_assignment(static_cast<Matrix&>(*this), m);
}
