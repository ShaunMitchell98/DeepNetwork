#include "Matrix.h"
#include "WeightMatrixGenerator.h"
#include <stdexcept>
#include <string>

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
				GenerateRandomWeights(Values.data(), rows * cols);
			}
	}

	double& Matrix::operator()(size_t row, size_t col) {
		if (row >= this->Rows) {
			throw "Row out of bounds";
		}
		else if (col > this->Cols) {
			throw "Col out of bounds";
		}

		return Values[row * Cols + col];
	}

	const double& Matrix::operator()(size_t row, size_t col) const {
		if (row >= this->Rows) {
			throw "Row out of bounds";
		}
		else if (col > this->Cols) {
			throw "Col out of bounds";
		}

		return Values[row * Cols + col];
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
						throw runtime_error("Cannot load matrix with non-constant column number.");
					}

					currentColNumber = 0;
					currentRowNumber++;
				}
			}

			Rows = currentRowNumber;
			Cols = expectedColNumber;
	}

	std::string Matrix::ToString() const {

		auto text = new std::string();
		char buffer[30];

		for (auto row = 0; row < Rows; row++) {
			for (auto col = 0; col < Cols; col++) {
				auto value = (*this)(row, col);
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

	void Matrix::Set(size_t rows, size_t cols, const double* values) {

		Rows = rows;
		Cols = cols;

		for (size_t i = 0; i < Rows; i++) {
			for (size_t j = 0; j < Cols; j++) {
				Values.push_back(*(values + (i * Cols) + j));
			}
		}
	}
}

