#include "CpuMatrix.h"

CpuMatrix::CpuMatrix()
#ifndef CPU_VECTOR
	: Matrix()
#endif
{}

//unique_ptr<Matrix> CpuMatrix::operator*(const Matrix& m) const {

//	auto c = unique_ptr<Matrix>(new CpuMatrix());
//	c->Initialise(Rows, m.GetCols(), false);

//	for (auto i = 0; i < Rows; i++) {
//		for (auto j = 0; j < m.GetCols(); j++) {
//			double tempValue = 0;
//			for (auto k = 0; k < Cols; k++) {
//				tempValue += (*this)(i, k) * m(k, j);
//			}

//			(*c)(j, i) = tempValue;
//		}
//	}

//	return move(c);
//}

//unique_ptr<Matrix> CpuMatrix::operator*(const double d) const {

//	auto c = unique_ptr<Matrix>(new CpuMatrix(*this));

//	for (auto i = 0; i < Rows; i++) {
//		for (auto j = 0; j < Cols; j++) {
//			(*c)(i, j) = (*this)(i, j) * d;
//		}
//	}

//	return move(c);
//}


//unique_ptr<Matrix> CpuMatrix::operator+(const Matrix& m) const {

//	auto c = unique_ptr<Matrix>(new CpuMatrix(*this));

//	for (auto i = 0; i < Rows; i++) {
//		for (auto j = 0; j < Cols; j++) {
//			(*c)(i, j) = (*this)(i, j) + m(i, j);
//		}
//	}

//	return move(c);
//}

//unique_ptr<Matrix> CpuMatrix::operator-(const Matrix& m) const {

//	auto c = unique_ptr<Matrix>(new CpuMatrix(*this));
//
//	for (auto i = 0; i < Rows; i++) {
//		for (auto j = 0; j < Cols; j++) {
//			(*c)(i, j) = (*this)(i, j) - m(i, j);
//		}
//	}

//	return move(c);
//}

//void CpuMatrix::operator+=(const Matrix& m) {

//	for (auto i = 0; i < m.GetRows(); i++) {
//		for (auto j = 0; j < m.GetCols(); j++) {
//			(*this)(i, j) = (*this)(i, j) + m(i, j);
//		}
//	}
//}

//unique_ptr<Matrix> CpuMatrix::operator~() const {

//	auto m = unique_ptr<Matrix>(new CpuMatrix());
//	m->Set(Cols, Rows, Values.data());
//	return move(m);
//}