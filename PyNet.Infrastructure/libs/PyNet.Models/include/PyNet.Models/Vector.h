#pragma once

#include "Matrix.h"
#include "Activation.h"
#include <memory>

namespace PyNet::Models {

	class Vector : public virtual Matrix
	{
	protected:
		std::shared_ptr<Activation> _activation;
	public:

		Vector(std::shared_ptr<Activation> activation);

		void Initialise(int rows, bool generateWeights) { Matrix::Initialise(rows, 1, generateWeights); }
		double GetValue(int row) const;
		double* GetAddress(int row) const;
		double* GetEnd() const;
		void SetValue(int row, double value);
		void ApplyActivation();
		virtual std::unique_ptr<Vector> CalculateActivationDerivative() = 0;
		void SetValue(double value);
		void AddValue(double value);
		void operator=(const Matrix& m);
		void operator=(const Vector& v);
		virtual void operator+=(const Vector& v) = 0;
		void Set(size_t rows, double* d);
		virtual std::unique_ptr<Vector> operator+(const Vector& v) = 0;
		virtual std::unique_ptr<Vector> operator-(const Vector& v) = 0;
		double operator|(const Vector& v) const;
		virtual std::unique_ptr<Vector> operator^(const Vector& v) = 0;
		virtual std::unique_ptr<Vector> operator/(const double d) = 0;
	};
}

