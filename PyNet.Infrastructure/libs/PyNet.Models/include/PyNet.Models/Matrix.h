#pragma once

#include <vector>
#include <string>
#include "Context.h"

namespace PyNet::Models {

	class Matrix {

	protected:
		int Rows = 0;
		int Cols = 0;
		std::vector<double> Values;
		di::Context& Context;

	public:

		Matrix(di::Context& context) : Context(context) {}
		void Initialise(size_t rows, size_t cols, bool generateWeights);
		double GetValue(size_t row, size_t col) const;
		void SetValue(size_t row, size_t col, double value);
		virtual int GetCols() const;
		virtual int GetRows() const;
		int GetSize() const;
		double* GetAddress(size_t row, size_t col);
		std::string ToString();
		void operator=(const Matrix& m);
		Matrix& operator~();
		Matrix& operator/(const double d);
		void Set(size_t rows, size_t cols, const double* values);
		virtual Matrix& operator*(const Matrix& m) const = 0;
		virtual Matrix& operator*(const double d) = 0;
		virtual Matrix& operator-(const Matrix& m) = 0;
		virtual void operator+=(const Matrix& m) = 0;
		std::vector<double> GetCValues() const;
		std::vector<double>& GetValues();
	};
}