#pragma once

#include "../matrix.h"
#include "../Activation Functions/activation_function.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void forward_propagate_layer(matrix weights, matrix inputLayer, matrix outputLayer, activation_function activationFunction);

#undef EXPORT_SYMBOL
}