#pragma once

#include <vector>

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void cuda_matrix_multiply(std::vector<double> A, std::vector<double>  B, std::vector<double> C, int Acols, int Bcols);

#undef EXPORT_SYMBOL
}