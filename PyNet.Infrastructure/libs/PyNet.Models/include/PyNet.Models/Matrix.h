#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cstdlib> 
#include <exception>
#include <functional>

using namespace std;

namespace PyNet::Models {

	class Matrix {
	private:

		void GenerateRandomWeights(double* address, int count) {

			for (int i = 0; i < count; i++) {
				address[i] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX) * 1000);
			}
		}

		int Rows = 0;
		int Cols = 0;

	public:
		vector<double> Values;

		auto begin() noexcept {
			return Values.begin();
		}
		auto end() noexcept {
			return Values.end();
		}

		double& operator()(size_t row, size_t col);

		const double& operator()(size_t row, size_t col) const;

		void Initialise(size_t rows, size_t cols, bool generateWeights);

		virtual int GetCols() const { return Cols; }
		virtual int GetRows() const { return Rows; }

		int GetSize() const { return GetCols() * GetRows(); }

		double* GetAddress(size_t row, size_t col) { return &Values[row * Cols + col]; }

		string ToString() const;

		void operator=(const Matrix& m) {
			Rows = m.Rows;
			Cols = m.Cols;
			Values = m.Values;
		}

		unique_ptr<Matrix> operator/(const double d) const {
			return (*this) * (1 / d);
		}

		void Set(size_t rows, size_t cols, const double* values);
		void Load(string_view value);

		virtual const vector<double>& GetCValues() const {
			return Values;
		}

		virtual vector<double>& GetValues() {
			return Values;
		}

		double operator|(const Matrix& m) const {
			if (m.GetRows() != GetRows() || m.GetCols() != GetCols()) {
				throw "Cannot calculate dot product for matrices with different dimensions.";
			}

			double result = 0;

			for (auto i = 0; i < m.GetRows(); i++) {
				for (auto j = 0; j < m.GetCols(); j++) {
					result += (*this)(i, j) * m(i, j);
				}
			}

			return result;
		}

		unique_ptr<Matrix> operator~() const {
			auto m = this->Copy();
			m->Set(GetCols(), GetRows(), ((Matrix*)this)->GetValues().data());
			return m;
		}

		//Virtual Methods

		//Performs matrix multiplication between two matrices;
		virtual unique_ptr<Matrix> operator*(const Matrix& m) const = 0;
		virtual unique_ptr<Matrix> operator*(const double d) const = 0;
		virtual unique_ptr<Matrix> operator+(const double d) const = 0;
		virtual unique_ptr<Matrix> operator+(const Matrix& m) const = 0;
		virtual unique_ptr<Matrix> operator-(const Matrix& m) const = 0;
		virtual unique_ptr<Matrix> operator-() const = 0;
		virtual unique_ptr<Matrix> operator^(const Matrix& m) const = 0;
		virtual unique_ptr<Matrix> Reciprocal() const = 0;
		virtual unique_ptr<Matrix> Exp() const = 0;
		virtual void operator+=(const Matrix& m) = 0;
		virtual ~Matrix() = default;
		virtual unique_ptr<Matrix> Copy() const = 0;
		virtual unique_ptr<Matrix> Max(double input) const = 0;
		virtual unique_ptr<Matrix> Step() const = 0;
	};
}