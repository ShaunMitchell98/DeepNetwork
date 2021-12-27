#include "CudaVector.h"
#include "Matrix_Operations.h"

CudaVector::CudaVector(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation) : Vector(context, activation), CudaMatrix(context), Matrix(context) {}

CudaVector::CudaVector(const CudaVector& v) : CudaMatrix(v.Context), Vector(v.Context, v._activation), Matrix(v.Context) {}

std::unique_ptr<Vector> CudaVector::operator-(const Vector& v) {
	auto c = std::unique_ptr<Vector>(new CudaVector(*this));
	c->Initialise(GetRows(), false);
	matrix_subtract(this->operator const PyNet::Models::Matrix & (), v, *c);
	return std::move(c);
}
