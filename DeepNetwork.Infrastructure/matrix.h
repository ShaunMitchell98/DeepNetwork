#ifndef KERNEL_MATRIX
#define KERNEL_MATRIX

#include <vector>

class Matrix {

public:
	std::vector<double> Values;
	int Rows;
	int Cols;

	Matrix(int rows, int cols);
	Matrix(int rows, int cols, double* values);
};

#endif