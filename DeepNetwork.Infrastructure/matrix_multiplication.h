#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "matrix.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void matrixMultiply(matrix A, matrix B, matrix C);

#undef EXPORT_SYMBOL
}

#endif