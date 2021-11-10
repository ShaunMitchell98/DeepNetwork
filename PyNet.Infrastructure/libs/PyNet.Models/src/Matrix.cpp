#include "Matrix.h"
#include "WeightMatrixGenerator.h"

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

	int Matrix::GetRows() {
		return Rows;
	}

	int Matrix::GetCols() {
		return Cols;
	}

	double* Matrix::GetAddress(int row, int col) {
		return &Values[(size_t)(row * Cols + col)];
	}

	Matrix* Matrix::operator~() {

		auto m = new Matrix(*this);
		m->Rows = Cols;
		m->Cols = Rows;
		return m;
	}

	Matrix& Matrix::operator/(const double d) {

		return (*this) * (1 / d);
	}

	void Matrix::operator=(const Matrix& m) {
		Rows = m.Rows;
		Cols = m.Cols;
		Values = m.Values;
	}
}

