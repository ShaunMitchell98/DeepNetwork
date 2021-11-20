#pragma once

#include "Matrix.h"
#include "Activation.h"
#include <memory>

namespace PyNet::Models {

	class Vector : public Matrix
	{
	protected:
		Activation& _activation;
	public:

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
		virtual void operator+=(const Vector& v) = 0;
		Vector& operator=(const double* d);
		virtual Vector& operator-(const Vector& v) = 0;
		double operator|(const Vector& v);
		Vector& operator^(const Vector& v);
		virtual Vector& operator*(const double d) = 0;
		Vector& operator/(const double d);

	};
}

