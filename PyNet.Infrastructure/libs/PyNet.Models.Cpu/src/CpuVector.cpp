#include "CpuVector.h"

namespace PyNet::Models::Cpu {

	CpuVector::CpuVector(std::shared_ptr<PyNet::DI::Context> context, std::shared_ptr<Activation> activation) : Vector(context, activation), CpuMatrix(context), Matrix(context) {}

	CpuVector::CpuVector(const CpuVector& v) : CpuMatrix(v.Context), Vector(v.Context, v._activation), Matrix(v.Context) {}
}
