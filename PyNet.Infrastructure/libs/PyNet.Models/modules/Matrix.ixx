module;
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> 
export module PyNet.Models:Matrix;

using namespace std;

export namespace PyNet::Models {

	class Matrix {
	private:

		void GenerateRandomWeights(double* address, int count) {

			for (int i = 0; i < count; i++) {
				address[i] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX) * 1000);
			}
		}

	protected:
		int Rows = 0;
		int Cols = 0;

	public:
		vector<double> Values;

		double& operator()(size_t row, size_t col) {
			if (row >= this->Rows) {
				throw "Row out of bounds";
			}
			else if (col > this->Cols) {
				throw "Col out of bounds";
			}

			return Values[row * Cols + col];
		}

		const double& operator()(size_t row, size_t col) const {
			if (row >= this->Rows) {
				throw "Row out of bounds";
			}
			else if (col > this->Cols) {
				throw "Col out of bounds";
			}

			return Values[row * Cols + col];
		}

		void Initialise(size_t rows, size_t cols, bool generateWeights) {
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

		virtual int GetCols() const { return Cols; }
		virtual int GetRows() const { return Rows; }
		int GetSize() const { return GetCols() * GetRows(); }
		double* GetAddress(size_t row, size_t col) { return &Values[row * Cols + col]; }

		string ToString() const {
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

		void operator=(const Matrix& m) {
			Rows = m.Rows;
			Cols = m.Cols;
			Values = m.Values;
		}

		unique_ptr<Matrix> operator/(const double d) const {
			return std::move((*this) * (1 / d));
		}

		void Set(size_t rows, size_t cols, const double* values) {
			Rows = rows;
			Cols = cols;

			for (size_t i = 0; i < Rows; i++) {
				for (size_t j = 0; j < Cols; j++) {
					Values.push_back(*(values + (i * Cols) + j));
				}
			}
		}

		void Load(string_view value) {
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

		vector<double> GetCValues() const {
			return this->Values;
		}

		//Virtual Methods

		virtual unique_ptr<Matrix> operator*(const Matrix& m) const = 0;
		virtual unique_ptr<Matrix> operator*(const double d) const = 0;
		virtual unique_ptr<Matrix> operator+(const Matrix& m) const = 0;
		virtual unique_ptr<Matrix> operator-(const Matrix& m) const = 0;
		virtual unique_ptr<Matrix> operator~() const = 0;
		virtual void operator+=(const Matrix& m) = 0;
	};
}