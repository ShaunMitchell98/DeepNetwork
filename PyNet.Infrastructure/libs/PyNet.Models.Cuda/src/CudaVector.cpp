#include "CudaVector.h"
#include "Matrix_Operations.h"

CudaVector::CudaVector(shared_ptr<Activation> activation) : Vector(activation){}

CudaVector::CudaVector(const CudaVector& v) : Vector(v._activation) {}

unique_ptr<Vector> CudaVector::operator+(const Vector& v) const {
	auto c = unique_ptr<Vector>(new CudaVector(*this));
	c->Initialise(GetRows(), false);
	matrix_add(this->operator const Matrix & (), v, *c);
	return move(c);
}

unique_ptr<Vector> CudaVector::operator-(const Vector& v) const {
	auto c = unique_ptr<Vector>(new CudaVector(*this));
	c->Initialise(GetRows(), false);
	matrix_subtract(this->operator const Matrix & (), v, *c);
	return move(c);
}

unique_ptr<Vector> CudaVector::operator^(const Vector& v) const {

	auto c = unique_ptr<Vector>(new CudaVector(this->_activation));
	c->Initialise(v.GetRows(), false);

	for (auto i = 0; i < v.GetRows(); i++) {
		(*c)[i] = (*this)[i] * v[i];
	}

	return move(c);
}

unique_ptr<Vector> CudaVector::CalculateActivationDerivative() const {
	auto derivative = _activation->CalculateDerivative(*this);
	return move(unique_ptr<Vector>(new CudaVector(std::move(*derivative))));
}

unique_ptr<Vector> CudaVector::operator/(const double d) const {
	auto result = Matrix::operator/(d);
	return move(unique_ptr<Vector>(new CudaVector(std::move(*result))));
}