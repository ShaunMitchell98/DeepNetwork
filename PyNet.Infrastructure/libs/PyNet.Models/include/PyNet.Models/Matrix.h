#pragma once

#include <vector>
#include <string>
#include "PyNet.DI/Context.h"

namespace PyNet::Models {

	class Matrix {

	protected:
		int Rows = 0;
		int Cols = 0;

	public:
		std::vector<double> Values;
		double& operator()(size_t row, size_t col);
		const double& operator()(size_t row, size_t col) const;

		void Initialise(size_t rows, size_t cols, bool generateWeights);
		virtual int GetCols() const;
		virtual int GetRows() const;
		int GetSize() const;
		double* GetAddress(size_t row, size_t col);
		std::string ToString() const;
		void operator=(const Matrix& m);
		std::unique_ptr<Matrix> operator/(const double d) const;
		void Set(size_t rows, size_t cols, const double* values);
		void Load(std::string_view value);

		virtual std::unique_ptr<Matrix> operator*(const Matrix& m) const = 0;
		virtual std::unique_ptr<Matrix> operator*(const double d) const = 0;
		virtual std::unique_ptr<Matrix> operator+(const Matrix& m) const = 0;
		virtual std::unique_ptr<Matrix> operator-(const Matrix& m) const = 0;
		virtual std::unique_ptr<Matrix> operator~() const = 0;
		virtual void operator+=(const Matrix& m) = 0;
		std::vector<double> GetCValues() const;
	};
}