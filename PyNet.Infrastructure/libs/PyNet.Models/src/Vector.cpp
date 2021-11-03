#include "Vector.h"

namespace PyNet::Models {

	double Vector::GetValue(int row) const {
		return ((Matrix*)(this))->GetValue(row, 0);
	}

	double* Vector::GetAddress(int row) const {
		return ((Matrix*)(this))->GetAddress(row, 0);
	}

	void Vector::SetValue(int row, double value) {
		return ((Matrix*)(this))->SetValue(row, 0, value);
	}

	void Vector::ApplyActivation() {

		if (_activation == NULL) {
			throw "Canot apply activation to vector, Activation is NULL";
		}
		
		_activation->Apply(this->Values);
	}

	void Vector::CalculateActivationDerivative(Vector* output) {
		_activation->CalculateDerivative(this, output);
	}

	double* Vector::GetEnd() const {
		return ((Matrix*)(this))->GetAddress(Rows - 1, 0);
	}

	void Vector::SetValue(double value) {

		for (auto i = 0; i < Rows; i++) {
			SetValue(i, value);
		}
	}

	void Vector::AddValue(double value) {
		for (auto i = 0; i < Rows; i++) {
			SetValue(i, GetValue(i) + value);
		}
	}

	void Vector::operator+=(const Vector& v) {
		Matrix::operator+=(v);
	}

	double Vector::operator|(const Vector& v) {

		if (v.Rows != this->Rows) {
			throw "Cannot calculate dot product for vectors with different lengths";
		}

		double result = 0;

		for (auto i = 0; i < v.Rows; i++) {
			result += this->GetValue(i) * v.GetValue(i);
		}

		return result;
	}

	Vector& Vector::operator^(const Vector& v) {

		auto c = new Vector(this->Rows, _cudaEnabled);

		for (auto i = 0; i < v.Rows; i++) {
			c->SetValue(i, this->GetValue(i) * v.GetValue(i));
		}

		return *c;
	}

	void Vector::operator=(const Matrix& m) {

		if (m.Cols != 1) {
			throw "Matrix cannot be converted to Vector";
		}

		Rows = m.Rows;
		Cols = m.Cols;
		Values = m.Values;
	}

	void Vector::operator=(const Vector& v) {
		operator=((Matrix&)v);
	}

	Vector& Vector::operator-(const Vector& v) {
		return static_cast<Vector&>((Matrix)(*this) - (Matrix)(v));
	}

	Vector& Vector::operator*(const double d) {
		return static_cast<Vector&>(Matrix::operator*(d));
	}

	Vector& Vector::operator/(const double d) {
		return static_cast<Vector&>(Matrix::operator/(d));
	}
}

