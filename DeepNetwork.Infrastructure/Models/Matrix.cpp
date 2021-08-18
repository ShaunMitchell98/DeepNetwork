#include "Matrix.h"
#include "../WeightMatrixGenerator.h"
#include <iterator>

namespace Models {

	Matrix::Matrix(int rows, int cols) {
		Values = std::vector<double>(rows * cols);
		generate_random_weights(Values.data(), rows * cols);
		Rows = rows;
		Cols = cols;
	}

	Matrix::Matrix(int rows, int cols, double* values) {
		Values = std::vector<double>(rows * cols);
		std::copy(&values[0], &values[rows * cols], Values.begin());
		Rows = rows;
		Cols = cols;
	}

	double Matrix::GetValue(int row, int col) {
		return Values[(size_t)(row * Cols + col)];
	}

	void Matrix::SetValue(int row, int col, double value) {
		Values[(size_t)row * Cols + col] = value;
	}

	double* Matrix::GetAddress(int row, int col) {
		return &Values[(size_t)(row * Cols + col)];
	}
}

