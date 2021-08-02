#pragma once

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void generate_random_weights(double* address, int count);

#undef EXPORT_SYMBOL
}