#pragma once
#include <memory>
#include "Matrix.h"
#include "Activation.h"

using namespace std;

namespace PyNet::Models {

	class Vector : public virtual Matrix
	{
	protected:
		shared_ptr<Activation> _activation;
	public:

		Vector(shared_ptr<Activation> activation) : _activation{ activation } {}

		void Initialise(int rows, bool generateWeights) { Matrix::Initialise(rows, 1, generateWeights); }

		const double& operator[](size_t row) const { return Values[row]; }

		double& operator[](size_t row) { return Values[row]; }

		double* GetAddress(int row) const { return ((Matrix*)(this))->GetAddress(row, 0); }

		void ApplyActivation() { _activation->Apply(*this); }

		void SetValue(double value) {
			for (auto i = 0; i < GetRows(); i++) {
				(*this)[i] = value;
			}
		}

		void AddValue(double value) {
			for (auto i = 0; i < GetRows(); i++) {
				(*this)[i] += value;
			}
		}
		void operator=(const Matrix& m) {
			if (m.GetCols() != 1) {
				throw "Matrix cannot be converted to Vector";
			}

			Matrix::operator=(m);
		}

		void operator=(const Vector& v) { operator=((Matrix&)v); }
	    double operator|(const Vector& v) const;

		void Set(size_t rows, double* d) {
			Initialise(rows, false);

			for (auto i = 0; i < rows; i++) {
				(*this)[i] = *(d + i);
			}
		}

		//Virtual Methods

		virtual unique_ptr<Vector> CalculateActivationDerivative() = 0;
		virtual unique_ptr<Vector> operator+(const Vector& v) const = 0;
		virtual void operator+=(const Vector& v) = 0;
		virtual unique_ptr<Vector> operator-(const Vector& v) const = 0;
		virtual unique_ptr<Vector> operator^(const Vector& v) const = 0;
		virtual unique_ptr<Vector> operator/(const double d) const = 0;
	};
}