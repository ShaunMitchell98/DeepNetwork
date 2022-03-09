module;
#include <memory>
export module PyNet.Models:Vector;

using namespace std;

import :Activation;
import :Matrix;

export namespace PyNet::Models {

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
			for (auto i = 0; i < Rows; i++) {
				(*this)[i] = value;
			}
		}

		void AddValue(double value) {
			for (auto i = 0; i < Rows; i++) {
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


		void Set(size_t rows, double* d) {
			Initialise(rows, false);

			for (auto i = 0; i < rows; i++) {
				(*this)[i] = *(d + i);
			}
		}

		double operator|(const Vector& v) const {
			if (v.Rows != this->Rows) {
				throw "Cannot calculate dot product for vectors with different lengths";
			}

			double result = 0;

			for (auto i = 0; i < v.Rows; i++) {
				result += (*this)[i] * v[i];
			}

			return result;
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

