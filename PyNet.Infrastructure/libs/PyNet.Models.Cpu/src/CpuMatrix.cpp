#include "CpuMatrix.h"

CpuMatrix::CpuMatrix()
#ifndef CPU_VECTOR
	: Matrix()
#endif
{}

std::unique_ptr<Matrix> CpuMatrix::operator*(const Matrix& m) const {

	auto c = std::unique_ptr<Matrix>(new CpuMatrix());
	c->Initialise(Rows, m.GetCols(), false);

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < m.GetCols(); j++) {
			double tempValue = 0;
			for (auto k = 0; k < Cols; k++) {
				tempValue += GetValue(i, k) * m.GetValue(k, j);
			}

			c->SetValue(j, i, tempValue);
		}
	}

	return std::move(c);
}

std::unique_ptr<Matrix> CpuMatrix::operator*(const double d) {

	auto c = std::unique_ptr<Matrix>(new CpuMatrix(*this));

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < Cols; j++) {
			c->SetValue(i, j, GetValue(i, j) * d);
		}
	}

	return std::move(c);
}


std::unique_ptr<Matrix> CpuMatrix::operator+(const Matrix& m) {

	auto c = std::unique_ptr<Matrix>(new CpuMatrix(*this));

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < Cols; j++) {
			c->SetValue(i, j, GetValue(i, j) + m.GetValue(i, j));
		}
	}

	return std::move(c);
}

std::unique_ptr<Matrix> CpuMatrix::operator-(const Matrix& m) {

	auto c = std::unique_ptr<Matrix>(new CpuMatrix(*this));

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < Cols; j++) {
			c->SetValue(i, j, GetValue(i, j) - m.GetValue(i, j));
		}
	}

	return std::move(c);
}

void CpuMatrix::operator+=(const Matrix& m) {

	for (auto i = 0; i < m.GetRows(); i++) {
		for (auto j = 0; j < m.GetCols(); j++) {
			this->SetValue(i, j, this->GetValue(i, j) + m.GetValue(i, j));
		}
	}
}

std::unique_ptr<Matrix> CpuMatrix::operator~() {

	auto m = std::unique_ptr<Matrix>(new CpuMatrix());
	m->Set(Cols, Rows, Values.data());
	return std::move(m);
}