#pragma once

#include <vector>

namespace PyNet::Models {

	class Matrix {

	private:
		bool _cudaEnabled;
	public:
		int Rows;
		int Cols;
		std::vector<double> Values;

		static auto factory() {
			return new Matrix(1, 2, true);
		}

		Matrix(int rows, int cols, bool cudaEnabled);
		Matrix(int rows, int cols, double* values, bool cudaEnabled);
		double GetValue(int row, int col) const;
		void SetValue(int row, int col, double value);
		double* GetAddress(int row, int col);
		Matrix& operator*(const Matrix& m);
	};
}

