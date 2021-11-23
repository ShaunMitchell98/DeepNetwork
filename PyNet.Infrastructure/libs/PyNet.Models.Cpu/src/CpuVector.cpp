#include "CpuVector.h"

CpuVector::CpuVector(di::Context& context, Activation& activation) : Vector(context, activation), CpuMatrix(context), Matrix(context) {}