#pragma once

#define CUDA_VECTOR

#include "PyNet.Models/Vector.h"
#include "CudaMatrix.h"

class __declspec(dllexport) CudaVector : public Vector, private CudaMatrix {
public:

	static auto factory(std::shared_ptr<Activation> activation) {
		return new CudaVector{ activation };
	}

	typedef Vector base;

	CudaVector(std::shared_ptr<Activation> activation);

	CudaVector(Matrix&& m) : Vector(nullptr) {
		static_cast<Vector*>(this)->Set(m.GetRows(), m.GetValues().data());
		m.Set(0, 0, nullptr);
	}

	std::unique_ptr<Matrix> operator*(const Matrix& m) const override {
		return CudaMatrix::operator*(m);
	}

	std::unique_ptr<Matrix> operator-(const Matrix& m) override {
		return CudaMatrix::operator-(m);
	}

	std::unique_ptr<Matrix> operator~() override {
		return CudaMatrix::operator~();
	}

	void operator+=(const Matrix& m) override {
		CudaMatrix::operator+=(m);
	}

	void operator+=(const Vector& v) override {
		return CudaMatrix::operator+=((Matrix&)v);
	}

	std::unique_ptr<Matrix> operator*(const double d) override {
		return CudaMatrix::operator*(d);
	}

	std::unique_ptr<Vector> CalculateActivationDerivative() override;

	std::unique_ptr<Vector> operator-(const Vector& v) override;

	std::unique_ptr<Vector> operator^(const Vector& v) override;

	std::unique_ptr<Vector> operator/(const double d) override;

	int GetRows() const override {
		return Vector::GetRows();
	}

	int GetCols() const override {
		return Vector::GetCols();
	}

	CudaVector(const CudaVector& v);

	operator const Matrix&() {
		auto& temp1 = static_cast<Vector&>(*this);
		auto& temp2 = static_cast<Matrix&>(temp1);
		return temp2;
	}
};

#undef CUDA_VECTOR