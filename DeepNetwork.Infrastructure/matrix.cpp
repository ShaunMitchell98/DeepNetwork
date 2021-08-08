#include "matrix.h"
#include "WeightMatrixGenerator.h"

Matrix::Matrix(int rows, int cols) {
	Values = std::vector<double>(rows * cols);
	generate_random_weights(Values.data(), rows * cols);

	Rows = rows;
	Cols = cols;
}

Matrix::Matrix(int rows, int cols, double* values) {
	Values = std::vector<double>(rows * cols);
	std::copy(&values[0], &values[rows * cols - 1], Values.data());
	Rows = rows;
	Cols = cols;
}