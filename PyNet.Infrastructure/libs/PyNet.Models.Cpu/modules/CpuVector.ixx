module;
#include <memory>
export module PyNet.Models.Cpu:CpuVector;

#define CPU_VECTOR

import PyNet.Models;
import :CpuMatrix;

using namespace std;

namespace PyNet::Models::Cpu {

	class CpuVector : public Vector, public CpuMatrix {

	public:
		static auto factory(shared_ptr<Activation> activation) {
			return new CpuVector{ activation };
		}

		typedef Vector base;

		CpuVector(shared_ptr<Activation> activation) : Vector(activation) {}

		CpuVector(Matrix&& m) : Vector(nullptr) {

			if (&this->operator Matrix &() == &m) {
				return;
			}

			Vector::Values = std::exchange(m.Values, vector<double>());
			Vector::Rows = m.GetRows();
		}

		unique_ptr<Matrix> operator*(const Matrix& m) const {
			return CpuMatrix::operator*(m);
		}

		unique_ptr<Matrix> operator+(const Matrix& m) const {
			return CpuMatrix::operator+(m);
		}

		unique_ptr<Matrix> operator-(const Matrix& m) const {
			return CpuMatrix::operator-(m);
		}

		void operator+=(const Matrix& m) {
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
			auto derivative = _activation->CalculateDerivative(this->operator const PyNet::Models::Matrix &());
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
			return std::unique_ptr<Vector>(dynamic_cast<Vector*>(CpuMatrix::operator+(v).get()));
		}

		unique_ptr<Vector> operator-(const Vector& v) const override {
			return std::unique_ptr<Vector>(dynamic_cast<Vector*>(CpuMatrix::operator-(v).get()));
		}

		unique_ptr<Vector> operator/(const double d) const override {
			auto result = Matrix::operator/(d);
			return move(unique_ptr<Vector>(new CpuVector(move(*result))));
		}

		int GetRows() const override {
			return Vector::GetRows();
		}

		int GetCols() const override {
			return Vector::GetCols();
		}

		operator Matrix& () {
			return static_cast<Matrix&>(static_cast<Vector&>(*this));
		}

		operator const Matrix& () {
			return static_cast<Matrix&>(static_cast<Vector&>(*this));
		}

		CpuVector(const CpuVector& v) : Vector(v._activation) {}
	};
}

#undef CPU_VECTOR