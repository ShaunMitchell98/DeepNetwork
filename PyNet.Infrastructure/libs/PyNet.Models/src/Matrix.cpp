#include "Matrix.h"
#include "WeightMatrixGenerator.h"

namespace PyNet::Models {

	void Matrix::Initialise(int rows, int cols) {
		Rows = rows;
		Cols = cols;
		generate_random_weights(Values.data(), rows * cols);
	}

	double Matrix::GetValue(size_t row, size_t col) const {

		if (row >= this->Rows) {
			throw "Row out of bounds";
		}
		else if (col > this->Cols) {
			throw "Col out of bounds";
		}

		return Values[row * Cols + col];
	}

	void Matrix::SetValue(int row, int col, double value) {
		Values[(size_t)row * Cols + col] = value;
	}

	int Matrix::GetRows() const {
		return Rows;
	}

	int Matrix::GetCols() const {
		return Cols;
	}

	int Matrix::GetSize() const {
		return Rows * Cols;
	}

	double* Matrix::GetAddress(size_t row, size_t col) {
		return &Values[row * Cols + col];
	}

	std::string Matrix::ToString() {

		auto text = new std::string();

		for (auto row = 0; row < Rows; row++) {
			for (auto col = 0; col < Cols; col++) {
				*text += std::to_string(GetValue(row, col));

				if (col != Cols - 1) {
					*text += ", ";
				}
			}

			*text += "\n";
		}

		return *text;
	}

	Matrix& Matrix::operator~() {

		auto m = Context.get<Matrix>();
		m.Rows = Cols;
		m.Cols = Rows;
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

	void Matrix::operator=(const double* m) {

		for (size_t i = 0; i < Rows; i++) {
			for (size_t j = 0; j < Cols; j++) {
				SetValue(i, j, *(m + (i * Cols) + j));
			}
		}
	}

	std::vector<double> Matrix::GetValues() const {
		return this->Values;
	}
}

