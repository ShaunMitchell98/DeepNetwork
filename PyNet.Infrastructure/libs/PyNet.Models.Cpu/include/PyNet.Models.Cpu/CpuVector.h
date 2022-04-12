#pragma once
#include <memory>
#include "PyNet.Models/Vector.h"
#include "PyNet.Models/Matrix.h"
#include "CpuMatrix.h"

#define CPU_VECTOR 

using namespace std;

namespace PyNet::Models::Cpu {

	class CpuVector : public Vector, public CpuMatrix {

	public:
		static auto factory(shared_ptr<Activation> activation) {
			return new CpuVector{ activation };
		}

		CpuVector(shared_ptr<Activation> activation) : Vector(activation) {}

		CpuVector(Matrix&& m) : Vector(nullptr) {
			static_cast<Vector*>(this)->Set(m.GetRows(), m.Values.data());
			m.Set(0, 0, nullptr);
		}

		unique_ptr<Matrix> operator*(const Matrix& m) const override {
			return CpuMatrix::operator*(m);
		}

		unique_ptr<Matrix> operator+(const Matrix& m) const override {
			return CpuMatrix::operator+(m);
		}

		unique_ptr<Matrix> operator-(const Matrix& m) const override {
			return CpuMatrix::operator-(m);
		}

		void operator+=(const Matrix& m) override {
			CpuMatrix::operator+=(m);
		}

		void operator+=(const Vector& v) override {
			return CpuMatrix::operator+=(v);
		}

		unique_ptr<Matrix> operator~() const override {
			return CpuMatrix::operator~();
		}

		unique_ptr<Matrix> operator*(const double d) const override {
			return CpuMatrix::operator*(d);
		}

		unique_ptr<Vector> CalculateActivationDerivative() override {
			auto derivative = _activation->CalculateDerivative(static_cast<Matrix&>(static_cast<Vector&>(*this)));
			return move(unique_ptr<Vector>(new CpuVector(move(*derivative))));
		}

		unique_ptr<Vector> operator^(const Vector& v) const override {
			auto c = unique_ptr<Vector>(new CpuVector(this->_activation));
			c->Initialise(v.GetRows(), false);

			for (auto i = 0; i < v.GetRows(); i++) {
				(*c)[i] = (*this)[i] * v[i];
			}

			return move(c);
		}

		unique_ptr<Vector> operator+(const Vector& v) const override {
			return unique_ptr<Vector>(dynamic_cast<Vector*>(CpuMatrix::operator+(v).get()));
		}

		unique_ptr<Vector> operator-(const Vector& v) const override {

			auto c = unique_ptr<Vector>(new CpuVector(nullptr));
			c->Initialise(GetRows(), false);

			for (auto i = 0; i < GetRows(); i++) {
				(*c)[i] = (*this)[i] - v[i];
			}

			return move(c);
		}

		unique_ptr<Vector> operator/(const double d) const override {
			auto result = CpuMatrix::operator/(d);
			return move(unique_ptr<Vector>(new CpuVector(move(*result))));
		}

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


		CpuVector(const CpuVector& v) : Vector(v._activation) {}

		~CpuVector() override = default;
	};
}

#undef CPU_VECTOR