module;
#include <memory>
#include "Matrix_Operations.h"
export module PyNet.Models.Cuda:CudaMatrix;

import PyNet.Models;

using namespace PyNet::Models;
using namespace std;

export namespace PyNet::Models::Cuda {
	class __declspec(dllexport) CudaMatrix
#ifdef CUDA_VECTOR
		: public virtual Matrix
#else
		: public Matrix
#endif
	{
	public:

		static auto factory() {
			return new CudaMatrix();
		}

		typedef Matrix base;

		CudaMatrix()
	#ifndef CUDA_VECTOR
			: Matrix()
	#endif
		{}

		unique_ptr<Matrix> operator*(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(Rows, m.GetCols(), false);
			cuda_matrix_multiply(this->GetCValues(), m.GetCValues(), c->Values, this->Rows, this->Cols, m.GetCols());

			return move(c);
		}

		unique_ptr<Matrix> operator*(const double d) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(Rows, Cols, false);
			multiply_matrix_and_double(this->GetCValues(), d, c->Values, this->GetRows(), this->GetCols());
			return move(c);
		}

		unique_ptr<Matrix> operator+(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(Rows, Cols, false);
			matrix_add(this->GetCValues(), m.GetCValues(), c->Values, this->Rows, this->Cols);
			return move(c);
		}

		unique_ptr<Matrix> operator-(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CudaMatrix());
			c->Initialise(Rows, Cols, false);
			matrix_subtract(this->GetCValues(), m.GetCValues(), c->Values, this->Rows, this->Cols);
			return move(c);
		}

		unique_ptr<Matrix> operator~() const override {
			auto m = unique_ptr<Matrix>(new CudaMatrix());
			m->Set(Cols, Rows, Values.data());
			return move(m);
		}

		void operator+=(const Matrix& m) override {
			matrix_addition_assignment(this->Values, m.GetCValues(), this->GetRows(), this->GetCols());
		}
	};
}
