#ifndef KERNEL_NORMALISE_LAYER
#define KERNEL_NORMALISE_LAYER

#include "matrix.h"
#include "network.h"
#include "Activation Functions/activation_function.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void normalise_layer(matrix A);

#undef EXPORT_SYMBOL
}

#endif