#include "CudaMatrix.h"
#include "Matrix_Operations.h"

CudaMatrix::CudaMatrix()
#ifndef CUDA_VECTOR
	: Matrix()
#endif
{}

unique_ptr<Matrix> CudaMatrix::operator*(const Matrix& m) const {
	auto c = unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, m.GetCols(), false);
	cuda_matrix_multiply(*this, m, *c);

	return move(c);
}

unique_ptr<Matrix> CudaMatrix::operator*(const double d) const {

	auto c = unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, Cols, false);
	multiply_matrix_and_double(*this, d, *c);
	return move(c);
}

unique_ptr<Matrix> CudaMatrix::operator+(const Matrix& m) const {
	auto c = unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, Cols, false);
	matrix_add(*this, m, *c);
	return move(c);
}

unique_ptr<Matrix> CudaMatrix::operator-(const Matrix& m) const {
	auto c = unique_ptr<Matrix>(new CudaMatrix());
	c->Initialise(Rows, Cols, false);
	matrix_subtract(*this, m, *c);
	return move(c);
 }

void CudaMatrix::operator+=(const Matrix& m) {

	matrix_addition_assignment(static_cast<Matrix&>(*this), m);
}


unique_ptr<Matrix> CudaMatrix::operator~() const {

	auto m = unique_ptr<Matrix>(new CudaMatrix());
	m->Set(Cols, Rows, Values.data());
	return move(m);
}
