#include "CpuMatrix.h"

Matrix& CpuMatrix::operator*(const Matrix& m) {

	auto c = new CpuMatrix(Context.get<CpuMatrix>());
	c->Initialise(Rows, m.GetCols());

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < m.GetCols(); j++) {
			double tempValue = 0;
			for (auto k = 0; k < Cols; k++) {
				tempValue += GetValue(i, k) * m.GetValue(k, j);
			}

			c->SetValue(j, i, tempValue);
		}
	}

	return *c;
}

Matrix& CpuMatrix::operator*(const double d) {

	auto c = new CpuMatrix(*this);

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < Cols; j++) {
			c->SetValue(i, j, GetValue(i, j) * d);
		}
	}

	return *c;
}

Matrix& CpuMatrix::operator-(const Matrix& m) {

	auto c = new CpuMatrix(*this);

	for (auto i = 0; i < Rows; i++) {
		for (auto j = 0; j < Cols; j++) {
			c->SetValue(i, j, GetValue(i, j) - m.GetValue(i, j));
		}
	}

	return *c;
}

void CpuMatrix::operator+=(const Matrix& m) {

	for (auto i = 0; i < m.GetRows(); i++) {
		for (auto j = 0; j < m.GetCols(); j++) {
			this->SetValue(i, j, this->GetValue(i, j) + m.GetValue(i, j));
		}
	}
}