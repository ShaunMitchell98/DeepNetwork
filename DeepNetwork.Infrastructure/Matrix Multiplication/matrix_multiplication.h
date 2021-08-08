#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "../matrix.h"

extern "C" {

	void matrix_multiply(Matrix* A, Matrix* B, Matrix* C);

}

#endif