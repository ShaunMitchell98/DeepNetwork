#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "../matrix.h"
#include "../network.h"
#include "../Activation Functions/activation_function.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void matrix_multiply(matrix A, matrix B, matrix C);

#undef EXPORT_SYMBOL
}

#endif