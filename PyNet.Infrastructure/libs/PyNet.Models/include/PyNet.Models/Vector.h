#pragma once

#include "Matrix.h"
#include "Activation.h"
#include <memory>
#include "Logistic.h"

using namespace ActivationFunctions;

namespace PyNet::Models {

	class Vector : public Matrix
	{
	private:
		std::unique_ptr<Activation> _activation;
	public:
		Vector(int rows, ActivationFunctionType activationFunctionType) : Vector(rows) {
			if (activationFunctionType == ActivationFunctionType::Logistic) {
				_activation = std::make_unique<Logistic>();
			}
		}
		Vector(int rows, double* values, ActivationFunctionType activationFunctionType) : Matrix(rows, 1, values) {
			if (activationFunctionType == ActivationFunctionType::Logistic) {
				_activation = std::make_unique<Logistic>();
			}
		}

		Vector(int rows) : Matrix(rows, 1) {}

		double GetValue(int row) const;
		double* GetAddress(int row) const;
		double* GetEnd() const;
		void SetValue(int row, double value);
		void ApplyActivation();
		void CalculateActivationDerivative(Vector* output);
		void SetValue(double value);
		void AddValue(double value);
		void operator=(const Matrix& m);
		void operator=(const Vector& v);
		void operator+=(const Vector& v);
		Vector& operator-(const Vector& v);
		double operator|(const Vector& v);
		Vector& operator^(const Vector& v);
		Vector& operator*(const double d);
		Vector& operator/(const double d);

	};
}

