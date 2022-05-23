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

			for (auto i = 1; i <= GetRows(); i++) {
				for (auto j = 1; j <= m.GetCols(); j++) {
					double tempValue = 0;
					for (auto k = 1; k <= GetCols(); k++) {
						tempValue += (*this)(i, k) * m(k, j);
					}

					(*c)(i, j) = tempValue;
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator+(const double d) const override {
			auto c = this->Copy();

			for (auto row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {
					(*c)(row, col) = (*this)(row, col) + d;
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator*(const double d) const override {
			auto c = this->Copy();

			for (auto row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {
					(*c)(row, col) = (*this)(row, col) * d;
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator+(const Matrix & m) const override {

			auto c = this->Copy();

			for (auto row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {
					(*c)(row, col) = (*this)(row, col) + m(row, col);
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator-(const Matrix & m) const override {
			auto c = this->Copy();

			for (auto row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {
					(*c)(row, col) = (*this)(row, col) - m(row, col);
				}
			}

			return c;
		}

		unique_ptr<Matrix> operator-() const override {
			auto c = this->Copy();

			for (auto row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {
					(*c)(row, col) = -(*this)(row, col);
				}
			}

			return c;
		}

		void operator+=(const Matrix & m) override {

			for (auto row = 1; row <= m.GetRows(); row++) {
				for (auto col = 1; col <= m.GetCols(); col++) {
					(*this)(row, col) = (*this)(row, col) + m(row, col);
				}
			}
		}

		unique_ptr<Matrix> Copy() const override {

			return unique_ptr<Matrix>(new CpuMatrix(*this));
		}

		unique_ptr<Matrix> Exp() const override {

			auto output = Copy();
				
			for (int row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {

					(*output)(row, col) = exp((*this)(row, col));
				}
			}

			return output;
		}

		unique_ptr<Matrix> Reciprocal() const override {

			auto output = Copy();

			for (int row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {

					(*output)(row, col) = 1 / (*this)(row, col);
				}
			}

			return output;
		}

		unique_ptr<Matrix> Max(double input) const override {

			auto output = Copy();

			for (int row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {

					(*output)(row, col) = max(input, (*this)(row, col));
				}
			}

			return output;
		}

		unique_ptr<Matrix> Step() const override {

			auto output = Copy();

			for (auto row = 1; row <= GetRows(); row++) {
				for (auto col = 1; col <= GetCols(); col++) {

					(*output)(row, col) = (*this)(row, col) <= 0 ? 0 : 1;
				}
			}

			return output;
		}


		unique_ptr<Matrix> operator^(const Matrix& m) const override {
			auto c = m.Copy();

			for (auto row = 1; row <= m.GetRows(); row++) {
				for (auto col = 1; col <= m.GetCols(); col++) {
					(*c)(row, col) = (*this)(row, col) * m(row, col);
				}
			}

			return c;
		}
	};
}