#include "CudaMatrix.h"
#include "Matrix_Operations.h"

namespace PyNet::Models::Cuda {

	unique_ptr<Matrix> CudaMatrix::operator*(const Matrix& m) const {
		auto c = unique_ptr<Matrix>(new CudaMatrix());
		c->Initialise(GetRows(), m.GetCols(), false);
		cuda_matrix_multiply(GetCValues(), m.GetCValues(), c->Values, GetRows(), GetCols(), m.GetCols());

		return move(c);
	}

	unique_ptr<Matrix> CudaMatrix::operator*(const double d) const {
		auto c = unique_ptr<Matrix>(new CudaMatrix());
		c->Initialise(GetRows(), GetCols(), false);
		multiply_matrix_and_double(GetCValues(), d, c->GetValues(), GetRows(), GetCols());
		return move(c);
	}

	unique_ptr<Matrix> CudaMatrix::operator+(const Matrix& m) const {
		auto c = unique_ptr<Matrix>(new CudaMatrix());
		c->Initialise(GetRows(), GetCols(), false);
		matrix_add(GetCValues(), m.GetCValues(), c->GetValues(), GetRows(), GetCols());
		return move(c);
	}

	unique_ptr<Matrix> CudaMatrix::operator-(const Matrix& m) const {
		auto c = unique_ptr<Matrix>(new CudaMatrix());
		c->Initialise(GetRows(), GetCols(), false);
		matrix_subtract(GetCValues(), m.GetCValues(), c->Values, GetRows(), GetCols());
		return move(c);
	}

	unique_ptr<Matrix> CudaMatrix::operator~() const {
		auto m = unique_ptr<Matrix>(new CudaMatrix());
		m->Set(GetCols(), GetRows(), ((Matrix*)this)->GetValues().data());
		return move(m);
	}

	void CudaMatrix::operator+=(const Matrix& m) {
		matrix_addition_assignment(GetValues(), m.GetCValues(), GetRows(), GetCols());
	}
}