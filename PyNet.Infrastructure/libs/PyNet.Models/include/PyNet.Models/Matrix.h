#pragma once

#include <vector>

namespace PyNet::Models {

	class Matrix {

	protected:
		bool _cudaEnabled;
	public:
		int Rows;
		int Cols;
		std::vector<double> Values;

		Matrix(int rows, int cols, bool cudaEnabled);
		Matrix(int rows, int cols, double* values, bool cudaEnabled);
		double GetValue(int row, int col) const;
		void SetValue(int row, int col, double value);
		double* GetAddress(int row, int col);
		void operator=(const Matrix& m);
		Matrix& operator*(const Matrix& m);
		Matrix* operator~();
		Matrix& operator*(const double d);
		Matrix& operator/(const double d);
		Matrix& operator-(const Matrix& m);
		void operator+=(const Matrix& m);
	};
}

