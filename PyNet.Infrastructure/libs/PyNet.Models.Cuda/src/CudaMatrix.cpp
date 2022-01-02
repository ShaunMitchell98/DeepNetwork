#include "CudaMatrix.h"
#include "Matrix_Operations.h"

CudaMatrix::CudaMatrix()
#ifndef CUDA_VECTOR
	: Matrix()
#endif
{}

std::unique_ptr<Matrix> CudaMatrix::operator*(const Matrix& m) const {
	auto c = std::unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, m.GetCols(), false);
	cuda_matrix_multiply(*this, m, *c);

	return std::move(c);
}

std::unique_ptr<Matrix> CudaMatrix::operator*(const double d) {

	auto c = std::unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, Cols, false);
	multiply_matrix_and_double(*this, d, *c);
	return std::move(c);
}

std::unique_ptr<Matrix> CudaMatrix::operator+(const Matrix& m) {
	auto c = std::unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, Cols, false);
	matrix_add(*this, m, *c);
	return std::move(c);
}

std::unique_ptr<Matrix> CudaMatrix::operator-(const Matrix& m) {
	auto c = std::unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, Cols, false);
	matrix_subtract(*this, m, *c);
	return std::move(c);
 }

void CudaMatrix::operator+=(const Matrix& m) {

	matrix_addition_assignment(static_cast<Matrix&>(*this), m);
}


std::unique_ptr<Matrix> CudaMatrix::operator~() {

	auto m = std::unique_ptr<Matrix>(new CudaMatrix());
	m->Set(Cols, Rows, Values.data());
	return std::move(m);
}
