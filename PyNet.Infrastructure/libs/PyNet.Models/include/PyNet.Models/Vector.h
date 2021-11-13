#pragma once

#include "Matrix.h"
#include "Activation.h"
#include <memory>

namespace PyNet::Models {

	class Vector : public Matrix
	{
	private:
		Activation& _activation;
	public:

		static auto factory(di::Context& context, Activation& activation) {
			return new Vector{ context, activation };
		}

		Vector(di::Context& context, Activation& activation);
		void Initialise(int rows) { Matrix::Initialise(rows, 1); }
		void SetActivationFunction(ActivationFunctionType activatonFunctionType);
		double GetValue(int row) const;
		double* GetAddress(int row) const;
		double* GetEnd() const;
		void SetValue(int row, double value);
		void ApplyActivation();
		void CalculateActivationDerivative(Vector& output);
		void SetValue(double value);
		void AddValue(double value);
		void operator=(const Matrix& m);
		void operator=(const Vector& v);
		void operator+=(const Vector& v);
		Vector& operator=(const double* v);
		Vector& operator-(const Vector& v);
		double operator|(const Vector& v);
		Vector& operator^(const Vector& v);
		Vector& operator*(const double d);
		Vector& operator/(const double d);

	};
}

