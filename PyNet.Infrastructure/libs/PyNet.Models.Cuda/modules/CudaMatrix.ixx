module;
#include <memory>
#include "Matrix_Operations.h"
export module PyNet.Models.Cuda:CudaMatrix;

import PyNet.Models;

using namespace PyNet::Models;
using namespace std;

export namespace PyNet::Models::Cuda {
	class __declspec(dllexport) CudaMatrix : public Matrix
	{
	public:

		static auto factory() {
			return new CudaMatrix();
		}

		CudaMatrix() : Matrix() {}

		unique_ptr<Matrix> operator*(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(GetRows(), m.GetCols(), false);
			cuda_matrix_multiply(GetCValues(), m.GetCValues(), c->Values, GetRows(), GetCols(), m.GetCols());

			return move(c);
		}

		unique_ptr<Matrix> operator*(const double d) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(GetRows(), GetCols(), false);
			multiply_matrix_and_double(GetCValues(), d, c->GetValues(), GetRows(), GetCols());
			return move(c);
		}

		unique_ptr<Matrix> operator+(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(GetRows(), GetCols(), false);
			matrix_add(GetCValues(), m.GetCValues(), c->GetValues(), GetRows(), GetCols());
			return move(c);
		}

		unique_ptr<Matrix> operator-(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(GetRows(), GetCols(), false);
			matrix_subtract(GetCValues(), m.GetCValues(), c->Values, GetRows(), GetCols());
			return move(c);
		}

		unique_ptr<Matrix> operator~() const override {
			auto m = unique_ptr<Matrix>(new CudaMatrix());
			m->Set(GetCols(), GetRows(), ((Matrix*)this)->GetValues().data());
			return move(m);
		}

		void operator+=(const Matrix& m) override {
			matrix_addition_assignment(GetValues(), m.GetCValues(), GetRows(), GetCols());
		}
	};
}
