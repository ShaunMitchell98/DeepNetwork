#include "CudaMatrix.h"
#include "Matrix_Operations.h"

CudaMatrix::CudaMatrix(std::shared_ptr<PyNet::DI::Context> context)
#ifndef CUDA_VECTOR
	: Matrix(context)
#endif
{}

std::unique_ptr<Matrix> CudaMatrix::operator*(const Matrix& m) const {
	auto c = Context->GetUnique<Matrix>();
	c->Initialise(Rows, m.GetCols(), false);
	cuda_matrix_multiply(*this, m, *c);

	return std::move(c);
}

std::unique_ptr<Matrix> CudaMatrix::operator*(const double d) {

	auto c = Context->GetUnique<Matrix>();
	c->Initialise(Rows, Cols, false);
	multiply_matrix_and_double(*this, d, *c);
	return std::move(c);
}

std::unique_ptr<Matrix> CudaMatrix::operator-(const Matrix& m) {
	auto c = Context->GetUnique<Matrix>();
	c->Initialise(Rows, Cols, false);
	matrix_subtract(*this, m, *c);
	return std::move(c);
 }

void CudaMatrix::operator+=(const Matrix& m) {

	matrix_addition_assignment(static_cast<Matrix&>(*this), m);
}
