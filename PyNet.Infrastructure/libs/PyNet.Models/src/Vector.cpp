#include "Vector.h"

namespace PyNet::Models {

	Vector::Vector(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation) : _activation { std::move(activation) } {}

	void Vector::SetActivationFunction(ActivationFunctionType activationFunctionType) {
		_activation = Context->GetShared<Activation>();
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
		
		_activation->Apply(*this);
	}

	std::unique_ptr<Vector> Vector::CalculateActivationDerivative() {
		return std::unique_ptr<Vector>(dynamic_cast<Vector*>(_activation->CalculateDerivative(*this).get()));
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

	double Vector::operator|(const Vector& v) const {

		if (v.Rows != this->Rows) {
			throw "Cannot calculate dot product for vectors with different lengths";
		}

		double result = 0;

		for (auto i = 0; i < v.Rows; i++) {
			result += this->GetValue(i) * v.GetValue(i);
		}

		return result;
	}

	std::unique_ptr<Vector> Vector::operator^(const Vector& v) {

		auto c = Context->GetUnique<Vector>();
		c->Initialise(v.Rows, false);

		for (auto i = 0; i < v.Rows; i++) {
			c->SetValue(i, this->GetValue(i) * v.GetValue(i));
		}

		return std::move(c);
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

	std::unique_ptr<Vector> Vector::operator/(const double d) {
		return std::unique_ptr<Vector>(dynamic_cast<Vector*>(Matrix::operator/(d).get()));
	}

	void Vector::Set(size_t rows, double* d) {

		Initialise(rows, false);

		for (auto i = 0; i < rows; i++) {
			Values[i] = *(d + i);
		}
	}
}

