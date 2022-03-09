module;
#include <memory>
#include "Matrix_Operations.h"
export module PyNet.Models.Cuda:CudaVector;

#define CUDA_VECTOR

import PyNet.Models;
import :CudaMatrix;

using namespace PyNet::Models;
using namespace std;

export namespace PyNet::Models::Cuda {
	class __declspec(dllexport) CudaVector : public Vector, private CudaMatrix {
	public:

		static auto factory(shared_ptr<Activation> activation) {
			return new CudaVector{ activation };
		}

		typedef Vector base;

		CudaVector(shared_ptr<Activation> activation) : Vector(activation) {}

		CudaVector(Matrix&& m) : Vector(nullptr) {
			static_cast<Vector*>(this)->Set(m.GetRows(), m.Values.data());
			m.Set(0, 0, nullptr);
		}

		unique_ptr<Matrix> operator*(const Matrix& m) const override {
			return CudaMatrix::operator*(m);
		}

		unique_ptr<Matrix> operator+(const Matrix& m) const override {
			return CudaMatrix::operator+(m);
		}

		unique_ptr<Matrix> operator-(const Matrix& m) const override {
			return CudaMatrix::operator-(m);
		}

		unique_ptr<Matrix> operator~() const override {
			return CudaMatrix::operator~();
		}

		void operator+=(const Matrix& m) override {
			CudaMatrix::operator+=(m);
		}

		void operator+=(const Vector& v) override {
			return CudaMatrix::operator+=((Matrix&)v);
		}

		unique_ptr<Matrix> operator*(const double d) const override {
			return CudaMatrix::operator*(d);
		}

		unique_ptr<Vector> CalculateActivationDerivative() override {
			auto derivative = _activation->CalculateDerivative(this->operator const Matrix & ());
			return std::move(unique_ptr<Vector>(new CudaVector(std::move(*derivative))));
		}

		unique_ptr<Vector> operator+(const Vector& v) const override {
			auto c = unique_ptr<Vector>(new CudaVector(*this));
			c->Initialise(GetRows(), false);
			matrix_add((this->operator const Matrix & ()).GetCValues(), v.GetCValues(), c->Values, this->GetRows(), this->GetCols());
			return std::move(c);
		}

		unique_ptr<Vector> operator-(const Vector& v) const override {
			auto c = unique_ptr<Vector>(new CudaVector(*this));
			c->Initialise(GetRows(), false);
			matrix_subtract((this->operator const Matrix & ()).GetCValues(), v.GetCValues(), c->Values, this->GetRows(), this->GetCols());
			return std::move(c);
		}

		unique_ptr<Vector> operator^(const Vector& v) const override {
			auto c = unique_ptr<Vector>(new CudaVector(this->_activation));
			c->Initialise(v.GetRows(), false);

			for (auto i = 0; i < v.GetRows(); i++) {
				(*c)[i] = (*this)[i] * v[i];
			}

			return std::move(c);
		}

		unique_ptr<Vector> operator/(const double d) const override {
			auto result = Matrix::operator/(d);
			return std::move(unique_ptr<Vector>(new CudaVector(std::move(*result))));
		}

		int GetRows() const override {
			return Vector::GetRows();
		}

		int GetCols() const override {
			return Vector::GetCols();
		}

		CudaVector(const CudaVector& v) : Vector(v._activation) {}

		operator const Matrix& () const {
			auto& temp1 = static_cast<const Vector&>(*this);
			auto& temp2 = static_cast<const Matrix&>(temp1);
			return temp2;
		}
	};
}


#undef CUDA_VECTOR