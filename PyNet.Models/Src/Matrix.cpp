#include "Matrix.h"
#include "PyNet.Infrastructure.Cuda/Matrix_Multiplication.h"
#include "WeightMatrixGenerator.h"
#include <iterator>

namespace PyNet::Models {

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

	double Matrix::GetValue(int row, int col) const {

		if (row >= this->Rows) {
			throw "Row out of bounds";
		}
		else if (col > this->Cols) {
			throw "Col out of bounds";
		}

		return Values[(size_t)(row * Cols + col)];
	}

	void Matrix::SetValue(int row, int col, double value) {
		Values[(size_t)row * Cols + col] = value;
	}

	double* Matrix::GetAddress(int row, int col) {
		return &Values[(size_t)(row * Cols + col)];
	}

	Matrix& Matrix::operator*(const Matrix& m) {

		/*auto C = new Matrix(A.Rows, B.Cols);
		for (auto i = 0; i < A.Rows; i++) {
			for (auto j = 0; j < B.Cols; j++) {
				double tempValue = 0;
				for (auto k = 0; k < A.Cols; k++) {
					tempValue += A.GetValue(i, k) * B.GetValue(k, j);
				}

				C->SetValue(j, i, tempValue);
			}
		}*/

		Matrix* c = new Matrix(this->Rows, m.Cols);
		cuda_matrix_multiply(this->Values, m.Values, c->Values, this->Cols, m.Cols);

		return *c;
	}
}

