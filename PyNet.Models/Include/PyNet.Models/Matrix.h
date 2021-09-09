#pragma once

#include <vector>

namespace Models {

	class Matrix {

	public:
		int Rows;
		int Cols;
		std::vector<double> Values;

		Matrix(int rows, int cols);
		Matrix(int rows, int cols, double* values);
		double GetValue(int row, int col);
		void SetValue(int row, int col, double value);
		double* GetAddress(int row, int col);
	};
}