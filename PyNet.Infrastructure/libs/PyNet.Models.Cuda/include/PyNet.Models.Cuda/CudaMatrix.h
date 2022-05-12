#pragma once
#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Models::Cuda {
	class __declspec(dllexport) CudaMatrix : public Matrix
	{
	public:

		static auto factory() {
			return new CudaMatrix();
		}

		CudaMatrix() : Matrix() {}

		CudaMatrix(const CudaMatrix& m) {
			Matrix::Initialise(m.GetRows(), m.GetCols(), false);
		}

		unique_ptr<Matrix> operator*(const Matrix& m) const override;

		unique_ptr<Matrix> operator*(const double d) const override;

		unique_ptr<Matrix> operator+(const Matrix& m) const override;

		unique_ptr<Matrix> operator+(const double d) const override;

		unique_ptr<Matrix> operator-(const Matrix& m) const override;

		unique_ptr<Matrix> operator-() const override;

		unique_ptr<Matrix> Exp() const override;

		unique_ptr<Matrix> Reciprocal() const override;

		unique_ptr<Matrix> Max(double input) const override;

		unique_ptr<Matrix> Step() const override;

		void operator+=(const Matrix& m) override;

		unique_ptr<Matrix> Copy() const override {

			return unique_ptr<Matrix>(new CudaMatrix(*this));
		}


		unique_ptr<Matrix> operator^(const Matrix& m) const override;

	};
}
