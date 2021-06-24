#ifndef KERNEL_CUH_
#define KERNEL_CUH_

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL void matrixMultiply(float* A, float* B, float* C, int N);

#undef EXPORT_SYMBOL
}

#endif