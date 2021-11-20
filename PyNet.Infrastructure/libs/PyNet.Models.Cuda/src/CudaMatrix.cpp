#include "CudaMatrix.h"
#include "Matrix_Operations.h"

Matrix& CudaMatrix::operator*(const Matrix& m) {
	auto& c = Context.get<Matrix>();
	c.Initialise(Rows, m.GetCols());

	cuda_matrix_multiply(this->Values, m.GetValues(), c.GetValues(), this->Cols, m.GetCols());

	return c;
}

Matrix& CudaMatrix::operator*(const double d) {

	auto& c = Context.get<Matrix>();

	multiply_matrix_and_double(this->Values, d, c.GetValues(), this->Cols, this->Rows);

	return c;
}

Matrix& CudaMatrix::operator-(const Matrix& m) {
	auto& c = Context.get<Matrix>();

	matrix_subtract(*this, m, c);

	return c;
}

void CudaMatrix::operator+=(const Matrix& m) {

	matrix_addition_assignment(*this, m);
}
