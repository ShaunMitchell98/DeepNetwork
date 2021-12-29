#include "Matrix.h"
#include "WeightMatrixGenerator.h"

namespace PyNet::Models {

	void Matrix::Initialise(size_t rows, size_t cols, bool generateWeights) {

		if (Rows > 0) {
			Values.resize(rows * cols);
		}
		else {
			Values = std::vector<double>(rows * cols);
		}

		Rows = rows;
		Cols = cols;

		if (generateWeights) {
			generate_random_weights(Values.data(), rows * cols);
		}
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

	void Matrix::SetValue(size_t row, size_t col, double value) {
		Values[row * Cols + col] = value;
	}

	int Matrix::GetRows() const {
		return Rows;
	}

	int Matrix::GetCols() const {
		return Cols;
	}

	int Matrix::GetSize() const {
		return GetCols() * GetRows();
	}

	double* Matrix::GetAddress(size_t row, size_t col) {
		return &Values[row * Cols + col];
	}

	void Matrix::Load(std::string_view value) {

		string currentValue;
		int expectedColNumber = 0;
		int currentRowNumber = 0;
		int currentColNumber = 0;
		Values = std::vector<double>();

		for (auto ch : value) {

			if (ch != ',' && ch != '\n') {
				currentValue += ch;
			}
			else {
				Values.push_back(stod(currentValue));
				currentValue.erase();
				currentColNumber++;
			}

			if (ch == '\n') {

				if (expectedColNumber == 0) {
					expectedColNumber = currentColNumber;
				}
				else if (expectedColNumber != currentColNumber) {
					throw std::exception("Cannot load matrix with non-constant column number.");
				}

				currentColNumber = 0;
				currentRowNumber++;
			}
		}

		Rows = currentRowNumber;
		Cols = expectedColNumber;
	}

	std::string Matrix::ToString() {

		auto text = new std::string();
		char buffer[30];

		for (auto row = 0; row < Rows; row++) {
			for (auto col = 0; col < Cols; col++) {
				auto value = GetValue(row, col);
				sprintf(buffer, "%.20f", value);
				*text += buffer;

				if (col != Cols - 1) {
					*text += ", ";
				}
			}

			*text += "\n";
		}

		return *text;
	}

	std::unique_ptr<Matrix> Matrix::operator/(const double d) {

		return std::move((*this) * (1 / d));
	}

	void Matrix::operator=(const Matrix& m) {
		Rows = m.Rows;
		Cols = m.Cols;
		Values = m.Values;
	}

	void Matrix::Set(size_t rows, size_t cols, const double* values) {

		Rows = rows;
		Cols = cols;

		for (size_t i = 0; i < Rows; i++) {
			for (size_t j = 0; j < Cols; j++) {
				Values.push_back(*(values + (i * Cols) + j));
			}
		}
	}

	std::vector<double> Matrix::GetCValues() const {
		return this->Values;
	}

	std::vector<double>& Matrix::GetValues() {
		return this->Values;
	}
}

