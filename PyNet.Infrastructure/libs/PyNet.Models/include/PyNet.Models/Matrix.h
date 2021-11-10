#pragma once

#include <vector>
#include <string>

namespace PyNet::Models {

	class Matrix {

	protected:
		int Rows;
		int Cols;
		std::vector<double> Values;

	public:

		Matrix(int rows, int cols);
		Matrix(int rows, int cols, double* values);
		double GetValue(int row, int col) const;
		void SetValue(int row, int col, double value);
		int GetCols() const;
		int GetRows() const;
		double* GetAddress(int row, int col);
		std::string ToString();
		void operator=(const Matrix& m);
		Matrix* operator~();
		Matrix& operator/(const double d);
		virtual Matrix& operator*(const Matrix& m);
		virtual Matrix& operator*(const double d);
		virtual Matrix& operator-(const Matrix& m);
		virtual void operator+=(const Matrix& m);
	};
}