#include "CudaVector.h"
#include "Matrix_Operations.h"

CudaVector::CudaVector(std::shared_ptr<Activation> activation) : Vector(activation){}

CudaVector::CudaVector(const CudaVector& v) : Vector(v._activation) {}

std::unique_ptr<Vector> CudaVector::operator+(const Vector& v) {
	auto c = std::unique_ptr<Vector>(new CudaVector(*this));
	c->Initialise(GetRows(), false);
	matrix_add(this->operator const PyNet::Models::Matrix & (), v, *c);
	return std::move(c);
}

std::unique_ptr<Vector> CudaVector::operator-(const Vector& v) {
	auto c = std::unique_ptr<Vector>(new CudaVector(*this));
	c->Initialise(GetRows(), false);
	matrix_subtract(this->operator const PyNet::Models::Matrix & (), v, *c);
	return std::move(c);
}

std::unique_ptr<Vector> CudaVector::operator^(const Vector& v) {

	auto c = std::unique_ptr<Vector>(new CudaVector(this->_activation));
	c->Initialise(v.GetRows(), false);

	for (auto i = 0; i < v.GetRows(); i++) {
		c->SetValue(i, this->GetValue(i) * v.GetValue(i));
	}

	return std::move(c);
}

std::unique_ptr<Vector> CudaVector::CalculateActivationDerivative() {
	auto derivative = _activation->CalculateDerivative(*this);
	return std::unique_ptr<Vector>(new CudaVector(std::move(*derivative)));
}

std::unique_ptr<Vector> CudaVector::operator/(const double d) {
	auto result = Matrix::operator/(d);
	return std::unique_ptr<Vector>(new CudaVector(std::move(*result)));
}