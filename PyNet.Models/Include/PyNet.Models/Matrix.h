#pragma once

#include <vector>

namespace Models {

	class Matrix {

	protected:
		std::vector<double> Values;

	public:
		int Rows;
		int Cols;

		Matrix(int rows, int cols);
		Matrix(int rows, int cols, double* values);
		double GetValue(int row, int col);
		void SetValue(int row, int col, double value);
		double* GetAddress(int row, int col);
	};
}