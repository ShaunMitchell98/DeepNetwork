#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "matrix.h"
#include "network.h"
#include "activation_function.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void matrix_multiply(matrix A, matrix B, matrix C);

	EXPORT_SYMBOL void forward_propagate_layer(matrix weights, matrix inputLayer, matrix outputLayer, activation_function activationFunction);

	EXPORT_SYMBOL float train_network(network network, matrix expectedLayer);

	void apply_logistic(matrix matrix);

#undef EXPORT_SYMBOL
}

#endif