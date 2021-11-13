#pragma once

#include <vector>
#include <string>
#include "Context.h"

namespace PyNet::Models {

	class Matrix {

	protected:
		int Rows = 0;
		int Cols = 0;
		std::vector<double> Values = std::vector<double>();
		di::Context& Context;

	public:

		Matrix(di::Context& context) : Context(context) {}
		void Initialise(int rows, int cols);
		double GetValue(size_t row, size_t col) const;
		void SetValue(int row, int col, double value);
		int GetCols() const;
		int GetRows() const;
		int GetSize() const;
		double* GetAddress(size_t row, size_t col);
		std::string ToString();
		void operator=(const Matrix& m);
		Matrix& operator~();
		Matrix& operator/(const double d);
		void operator=(const double* v);
		virtual Matrix& operator*(const Matrix& m) = 0;
		virtual Matrix& operator*(const double d) = 0;
		virtual Matrix& operator-(const Matrix& m) = 0;
		virtual void operator+=(const Matrix& m) = 0;
		std::vector<double> GetValues() const;
	};
}