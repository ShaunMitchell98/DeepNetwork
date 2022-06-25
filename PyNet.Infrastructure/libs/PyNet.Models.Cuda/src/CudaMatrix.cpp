#include "CudaMatrix.h"
#include "Matrix_Operations.h"
#include "cuda_runtime.h"

namespace PyNet::Models::Cuda {

	unique_ptr<Matrix> CudaMatrix::operator*(const Matrix& m) const {
		auto c = Copy();
		c->Initialise(GetRows(), m.GetCols(), false);
		matrix_multiply(GetCValues(), m.GetCValues(), c->Values, GetRows(), GetCols(), m.GetCols());

		return c;
	}

	unique_ptr<Matrix> CudaMatrix::operator*(const double d) const {
		auto c = Copy();
		multiply_matrix_and_double(GetCValues(), d, c->GetValues(), GetRows(), GetCols());
		return c;
	}

	unique_ptr<Matrix> CudaMatrix::operator+(const double d) const {
		auto c = Copy();
		add_matrix_and_double(GetCValues(), d, c->GetValues(), GetRows(), GetCols());
		return c;
	}

	unique_ptr<Matrix> CudaMatrix::operator+(const Matrix& m) const {
		auto c = Copy();
		matrix_add(GetCValues(), m.GetCValues(), c->GetValues(), GetRows(), GetCols());
		return c;
	}

	unique_ptr<Matrix> CudaMatrix::operator-(const Matrix& m) const {
		auto c = Copy();
		matrix_subtract(GetCValues(), m.GetCValues(), c->Values, GetRows(), GetCols());
		return c;
	}

	unique_ptr<Matrix> CudaMatrix::operator-() const {
		auto c = Copy();
		auto zeroMatrix = Copy();
		matrix_subtract(zeroMatrix->GetCValues(), GetCValues(), c->GetValues(), zeroMatrix->GetRows(), zeroMatrix->GetCols());
		return c;
	}

	void CudaMatrix::operator+=(const Matrix& m) {
		matrix_addition_assignment(GetValues(), m.GetCValues(), GetRows(), GetCols());
	}

	unique_ptr<Matrix> CudaMatrix::Exp() const {
		auto output = Copy();
		matrix_exp(GetCValues(), output->GetValues(), GetRows(), GetCols());
		return output;
	}

	unique_ptr<Matrix> CudaMatrix::Reciprocal() const {
		auto output = Copy();
		matrix_reciprocal(GetCValues(), output->GetValues(), GetRows(), GetCols());
		return output;
	}

	unique_ptr<Matrix> CudaMatrix::Max(double input) const {
		auto output = Copy();
		matrix_max(GetCValues(), input, output->GetValues(), GetRows(), GetCols());
		return output;
	}

	unique_ptr<Matrix> CudaMatrix::Step() const {
		auto output = Copy();
		matrix_step(GetCValues(), output->GetValues(), GetRows(), GetCols());
		return output;
	}

	unique_ptr<Matrix> CudaMatrix::operator^(const Matrix& m) const {
		auto output = m.Copy();
		matrix_hadamard(GetCValues(), m.GetCValues(), output->GetValues(), GetRows(), GetCols());
		return output;
	}
}