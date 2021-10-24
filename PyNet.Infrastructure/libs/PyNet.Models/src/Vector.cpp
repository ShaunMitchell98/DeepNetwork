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

	double Vector::CalculateActivationDerivative(double input) {
		return _activation->CalculateDerivative(input);
	}

	double* Vector::GetEnd() const {
		return ((Matrix*)(this))->GetAddress(Rows - 1, 0);
	}

	void Vector::operator=(Vector& v) {
		Rows = v.Rows;
		Cols = v.Cols;
		Values = v.Values;
	}

	void Vector::operator+=(const Vector& v) {

		for (auto i = 0; i < v.Rows; i++) {
			this->SetValue(i, this->GetValue(i) + v.GetValue(i));
		}
	}
}

