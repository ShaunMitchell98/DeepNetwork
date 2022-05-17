#pragma once
#include <memory>
#include "PyNet.Models/Matrix.h"
#include <algorithm>

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Models::Cpu {

	class CpuMatrix : public Matrix
	{
	public:

		static auto factory() {
			return new CpuMatrix();
		}

		CpuMatrix() : Matrix() {}

		CpuMatrix(const CpuMatrix & m) {
			Matrix::Initialise(m.GetRows(), m.GetCols(), false);
		}

		unique_ptr<Matrix> operator*(const Matrix & m) const override {
			auto c = this->Copy();
			c->Initialise(GetRows(), m.GetCols(), false);

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < m.GetCols(); j++) {
					double tempValue = 0;
					for (auto k = 0; k < GetCols(); k++) {
						tempValue += (*this)(i, k) * m(k, j);
					}

					(*c)(i, j) = tempValue;
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator+(const double d) const override {
			auto c = this->Copy();

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = (*this)(i, j) + d;
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator*(const double d) const override {
			auto c = this->Copy();

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = (*this)(i, j) * d;
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator+(const Matrix & m) const override {

			auto c = this->Copy();

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = (*this)(i, j) + m(i, j);
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator-(const Matrix & m) const override {
			auto c = this->Copy();

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = (*this)(i, j) - m(i, j);
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator-() const override {
			auto c = this->Copy();

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = -(*this)(i, j);
				}
			}

			return c;
		}

		void operator+=(const Matrix & m) override {

			for (auto i = 0; i < m.GetRows(); i++) {
				for (auto j = 0; j < m.GetCols(); j++) {
					(*this)(i, j) = (*this)(i, j) + m(i, j);
				}
			}
		}

		unique_ptr<Matrix> Copy() const override {

			return unique_ptr<Matrix>(new CpuMatrix(*this));
		}

		unique_ptr<Matrix> Exp() const override {

			auto output = Copy();
				
			for (int i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {

					(*output)(i, j) = exp((*this)(i, j));
				}
			}

			return output;
		}

		unique_ptr<Matrix> Reciprocal() const override {

			auto output = Copy();

			for (int i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {

					(*output)(i, j) = 1 / (*this)(i, j);
				}
			}

			return output;
		}

		unique_ptr<Matrix> Max(double input) const override {

			auto output = Copy();

			for (int i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {

					(*output)(i, j) = max(0.0, (*this)(i, j));
				}
			}

			return output;
		}

		unique_ptr<Matrix> Step() const override {

			auto output = Copy();

			for (int i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {

					(*output)(i, j) = (*this)(i, j) <= 0 ? 0 : 1;
				}
			}

			return output;
		}


		unique_ptr<Matrix> operator^(const Matrix& m) const override {
			auto c = m.Copy();

			for (auto row = 0; row < m.GetRows(); row++) {
				for (auto col = 0; col < m.GetCols(); col++) {
					(*c)(row, col) = (*this)(row, col) * m(row, col);
				}
			}

			return c;
		}
	};
}