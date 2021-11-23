#include "CudaVector.h"

CudaVector::CudaVector(di::Context& context, PyNet::Models::Activation& activation) : Vector(context, activation), CudaMatrix(context), Matrix(context) {}

CudaVector::CudaVector(const CudaVector& v) : CudaMatrix(v.Context), Vector(v.Context, v._activation), Matrix(v.Context) {}

Vector& CudaVector::operator*(const double d) {
	const auto& m = CudaMatrix::operator*(d);
	auto& v = Context.get<Vector>();
	v = m;
	return v;
}

Vector& CudaVector::operator-(const Vector& v) {
	const auto& m = CudaMatrix::operator-(v);
	auto& v2 = Context.get<Vector>();
	v2 = m;
	return v2;
}