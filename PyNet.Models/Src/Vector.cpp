#include "Vector.h"

namespace Models {

	double Vector::GetValue(int row) {
		return ((Matrix*)(this))->GetValue(row, 0);
	}

	double* Vector::GetAddress(int row) {
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

	double* Vector::GetEnd() {
		return ((Matrix*)(this))->GetAddress(Rows - 1, 0);
	}
}
