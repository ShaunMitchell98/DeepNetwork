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

		unique_ptr<Matrix> operator*(const Matrix& m) const override;

		unique_ptr<Matrix> operator*(const double d) const override;

		unique_ptr<Matrix> operator+(const Matrix& m) const override;

		unique_ptr<Matrix> operator-(const Matrix& m) const override;

		unique_ptr<Matrix> operator~() const override;

		void operator+=(const Matrix& m) override;
	};
}
