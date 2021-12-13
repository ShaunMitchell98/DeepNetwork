#pragma once

#define CPU_VECTOR

#include "PyNet.Models/Vector.h"
#include "CpuMatrix.h"

namespace PyNet::Models::Cpu {

	class CpuVector : public PyNet::Models::Vector, public CpuMatrix {

	public:
		static auto factory(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation) {
			return new CpuVector{ context, activation };
		}

		typedef PyNet::Models::Vector base;

		CpuVector(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation);

		std::unique_ptr<Matrix> operator*(const Matrix& m) const {
			return CpuMatrix::operator*(m);
		}

		std::unique_ptr<Matrix> operator-(const Matrix& m) {
			return CpuMatrix::operator-(m);
		}

		void operator+=(const Matrix& m) {
			CpuMatrix::operator+=(m);
		}

		void operator+=(const Vector& v) override {
			return CpuMatrix::operator+=(v);
		}

		std::unique_ptr<Matrix> operator*(const double d) override {
			return CpuMatrix::operator*(d);
		}

		std::unique_ptr<Vector> operator-(const Vector& v) override {
			return std::unique_ptr<Vector>(dynamic_cast<Vector*>(CpuMatrix::operator-(v).get()));
		}

		int GetRows() const override {
			return Vector::GetRows();
		}

		int GetCols() const override {
			return Vector::GetCols();
		}

		operator Matrix& () {
			return static_cast<PyNet::Models::Matrix&>(static_cast<Vector&>(*this));
		}

		CpuVector(const CpuVector& v);
	};
}


#undef CPU_VECTOR