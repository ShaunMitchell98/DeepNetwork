#include "Vector.h"

namespace PyNet::Models {

	Vector::Vector(std::shared_ptr<Activation> activation) : _activation { activation } {}

	const double& Vector::operator[](size_t row) const {
		return Values[row];
	}

	double& Vector::operator[](size_t row) {
		return Values[row];
	}

	double* Vector::GetAddress(int row) const {
		return ((Matrix*)(this))->GetAddress(row, 0);
	}

	void Vector::ApplyActivation() {
		
		_activation->Apply(*this);
	}

	void Vector::SetValue(double value) {

		for (auto i = 0; i < Rows; i++) {
			(*this)[i] = value;
		}
	}

	void Vector::AddValue(double value) {
		for (auto i = 0; i < Rows; i++) {
			(*this)[i] += value;
		}
	}

	double Vector::operator|(const Vector& v) const {

		if (v.Rows != this->Rows) {
			throw "Cannot calculate dot product for vectors with different lengths";
		}

		double result = 0;

		for (auto i = 0; i < v.Rows; i++) {
			result += (*this)[i] * v[i];
		}

		return result;
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

	void Vector::Set(size_t rows, double* d) {

		Initialise(rows, false);

		for (auto i = 0; i < rows; i++) {
			(*this)[i] = *(d + i);
		}
	}
}

