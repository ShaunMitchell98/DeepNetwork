#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"

using namespace PyNet::Models;
using namespace std;

class CpuMatrix 
#ifdef CPU_VECTOR
	: public virtual Matrix
#else
	: public Matrix
#endif
{
public:

	static auto factory() {
		return new CpuMatrix();
	}

	typedef Matrix base;

	CpuMatrix();
	const double& operator()(size_t row, size_t col) const { return Matrix::operator()(row, col); }
	double& operator()(size_t row, size_t col) { return Matrix::operator()(row, col); }

	unique_ptr<Matrix> operator*(const Matrix& m) const override {
		auto c = unique_ptr<Matrix>(new CpuMatrix());
		c->Initialise(Rows, m.GetCols(), false);

		for (auto i = 0; i < Rows; i++) {
			for (auto j = 0; j < m.GetCols(); j++) {
				double tempValue = 0;
				for (auto k = 0; k < Cols; k++) {
					tempValue += (*this)(i, k) * m(k, j);
				}

				(*c)(j, i) = tempValue;
			}
		}

		return move(c);
	}

	unique_ptr<Matrix> operator*(const double d) const override {
			auto c = unique_ptr<Matrix>(new CpuMatrix(*this));

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < Cols; j++) {
			(*c)(i, j) = (*this)(i, j) * d;
		}
	}

	return move(c);
	}
	unique_ptr<Matrix> operator+(const Matrix& m) const override {
		auto c = unique_ptr<Matrix>(new CpuMatrix(*this));

		for (auto i = 0; i < Rows; i++) {
			for (auto j = 0; j < Cols; j++) {
				(*c)(i, j) = (*this)(i, j) + m(i, j);
			}
		}

		return move(c);
	}

	unique_ptr<Matrix> operator-(const Matrix& m) const override {
		auto c = unique_ptr<Matrix>(new CpuMatrix(*this));

		for (auto i = 0; i < Rows; i++) {
			for (auto j = 0; j < Cols; j++) {
				(*c)(i, j) = (*this)(i, j) - m(i, j);
			}
		}

		return move(c);
	}

	unique_ptr<Matrix> operator~() const override{
		auto m = unique_ptr<Matrix>(new CpuMatrix());
		m->Set(Cols, Rows, Values.data());
		return move(m);
	}

	void operator+=(const Matrix& m) override {
		for (auto i = 0; i < m.GetRows(); i++) {
			for (auto j = 0; j < m.GetCols(); j++) {
				(*this)(i, j) = (*this)(i, j) + m(i, j);
			}
		}
	}
};