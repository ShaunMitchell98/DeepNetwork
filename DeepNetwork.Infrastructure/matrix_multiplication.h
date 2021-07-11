#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "matrix.h"
#include "network.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void matrix_multiply(matrix A, matrix B, matrix C);

	EXPORT_SYMBOL float train_network(network network, matrix expectedLayer);

#undef EXPORT_SYMBOL
}

#endif