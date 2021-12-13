#include "CudaVector.h"

CudaVector::CudaVector(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation) : Vector(context, activation), CudaMatrix(context), Matrix(context) {}

CudaVector::CudaVector(const CudaVector& v) : CudaMatrix(v.Context), Vector(v.Context, v._activation), Matrix(v.Context) {}