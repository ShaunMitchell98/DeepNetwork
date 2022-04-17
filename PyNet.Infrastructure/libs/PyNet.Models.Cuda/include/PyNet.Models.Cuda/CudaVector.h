#pragma once
#include <memory>
#include "PyNet.Models/Vector.h"
#include "CudaMatrix.h"

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Models::Cuda {
	class __declspec(dllexport) CudaVector : public Vector, private CudaMatrix {
	public:

		static auto factory(shared_ptr<Activation> activation) {
			return new CudaVector{ activation };
		}

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

		unique_ptr<Vector> CalculateActivationDerivative() override;

		unique_ptr<Vector> operator+(const Vector& v) const override;

		unique_ptr<Vector> operator-(const Vector& v) const override;

		unique_ptr<Vector> operator^(const Vector& v) const override;

		unique_ptr<Vector> operator/(const double d) const override;

		int GetRows() const override {
			return Vector::GetRows();
		}

		int GetCols() const override {
			return Vector::GetCols();
		}

		vector<double>& GetValues() override {
			return Vector::GetValues();
		}

		const vector<double>& GetCValues() const override {
			return Vector::GetCValues();
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