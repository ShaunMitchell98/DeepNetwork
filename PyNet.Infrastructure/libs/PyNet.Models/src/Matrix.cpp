#include "Matrix.h"
#include "PyNet.Infrastructure.Cuda/Matrix_Multiplication.h"
#include "WeightMatrixGenerator.h"
#include <iterator>

namespace PyNet::Models {

	Matrix::Matrix(int rows, int cols, bool cudaEnabled) {
		Values = std::vector<double>(rows * cols);
		generate_random_weights(Values.data(), rows * cols);
		Rows = rows;
		Cols = cols;
		_cudaEnabled = cudaEnabled;
	}

	Matrix::Matrix(int rows, int cols, double* values, bool cudaEnabled) {
		Values = std::vector<double>(rows * cols);
		std::copy(&values[0], &values[rows * cols], Values.begin());
		Rows = rows;
		Cols = cols;
		_cudaEnabled = cudaEnabled;
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

	Matrix* Matrix::operator~() {

		auto m = new Matrix(*this);
		m->Rows = Cols;
		m->Cols = Rows;
		return m;
	}

	Matrix& Matrix::operator*(const Matrix& m) {

		auto c = new Matrix(this->Rows, m.Cols, _cudaEnabled);

		if (_cudaEnabled) {
			Matrix* c = new Matrix(this->Rows, m.Cols, _cudaEnabled);
			cuda_matrix_multiply(this->Values, m.Values, c->Values, this->Cols, m.Cols);
		}

		else {
			for (auto i = 0; i < this->Rows; i++) {
				for (auto j = 0; j < m.Cols; j++) {
					double tempValue = 0;
					for (auto k = 0; k < this->Cols; k++) {
						tempValue += this->GetValue(i, k) * m.GetValue(k, j);
					}

					c->SetValue(j, i, tempValue);
				}
			}
		}

		return *c;
	}

	Matrix& Matrix::operator*(const double d) {

		auto c = new Matrix(Rows, Cols, _cudaEnabled);

		for (auto i = 0; i < this->Rows; i++) {
			for (auto j = 0; j < this->Cols; j++) {
				c->SetValue(i, j, this->GetValue(i, j) * d);
			}
		}

		return *c;
	}

	Matrix& Matrix::operator-(const Matrix& m) {

		auto c = new Matrix(Rows, Cols, _cudaEnabled);

		for (auto i = 0; i < this->Rows; i++) {
			for (auto j = 0; j < this->Cols; j++) {
				c->SetValue(i, j, this->GetValue(i, j) - m.GetValue(i, j));
			}
		}

		return *c;
	}

	Matrix& Matrix::operator/(const double d) {

		return (*this) * (1 / d);
	}

	void Matrix::operator=(const Matrix& m) {
		Rows = m.Rows;
		Cols = m.Cols;
		Values = m.Values;
	}

	void Matrix::operator+=(const Matrix& m) {
		for (auto i = 0; i < m.Rows; i++) {
			for (auto j = 0; j < m.Cols; j++) {
				this->SetValue(i, j, this->GetValue(i, j) + m.GetValue(i, j));
			}
		}
	}
}

