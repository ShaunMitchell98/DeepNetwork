#include "Vector.h"

namespace PyNet::Models {

	Vector::Vector(di::Context& context, Activation& activation) : _activation { activation } {}

	void Vector::SetActivationFunction(ActivationFunctionType activationFunctionType) {
		_activation = Context.get<Activation>();
	}

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

		//if (_activation == NULL) {
		//	throw "Canot apply activation to vector, Activation is NULL";
		//}
		
		_activation.Apply(*this);
	}

	void Vector::CalculateActivationDerivative(Vector& output) {
		_activation.CalculateDerivative(*this, output);
	}

	double* Vector::GetEnd() const {
		return ((Matrix*)(this))->GetAddress(static_cast<size_t>(Rows) - 1, 0);
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

		auto& c = Context.get<Vector>();
		c.Initialise(v.Rows, false);

		for (auto i = 0; i < v.Rows; i++) {
			c.SetValue(i, this->GetValue(i) * v.GetValue(i));
		}

		return c;
	}

	void Vector::operator=(const Matrix& m) {

		if (m.GetCols() != 1) {
			throw "Matrix cannot be converted to Vector";
		}

		Matrix::operator=(m);
	}

	void Vector::operator=(const Vector& v) {
		operator=((Matrix&)v);
	}

	Vector& Vector::operator/(const double d) {
		return dynamic_cast<Vector&>(Matrix::operator/(d));
	}

	void Vector::Set(size_t rows, double* d) {

		Initialise(rows, false);

		for (auto i = 0; i < rows; i++) {
			Values[i] = *(d + i);
		}
	}
}

