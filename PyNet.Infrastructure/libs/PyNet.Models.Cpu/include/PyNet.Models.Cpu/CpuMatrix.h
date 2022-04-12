#pragma once
#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Models::Cpu {

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


		unique_ptr<Matrix> operator*(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CpuMatrix());
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

			return move(c);
		}

		unique_ptr<Matrix> operator*(const double d) const override {
			auto c = unique_ptr<Matrix>(new CpuMatrix());
			c->Initialise(GetRows(), GetCols(), false);

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = (*this)(i, j) * d;
				}
			}

			return move(c);
		}

		unique_ptr<Matrix> operator+(const Matrix& m) const override {

			auto c = unique_ptr<Matrix>(new CpuMatrix());
			c->Initialise(GetRows(), GetCols(), false);

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = (*this)(i, j) + m(i, j);
				}
			}

			return move(c);
		}

		unique_ptr<Matrix> operator-(const Matrix& m) const override {
			auto c = unique_ptr<Matrix>(new CpuMatrix());
			c->Initialise(GetRows(), GetCols(), false);

			for (auto i = 0; i < GetRows(); i++) {
				for (auto j = 0; j < GetCols(); j++) {
					(*c)(i, j) = (*this)(i, j) - m(i, j);
				}
			}

			return move(c);
		}

		unique_ptr<Matrix> operator~() const override {
			auto m = unique_ptr<Matrix>(new CpuMatrix());
			m->Set(GetCols(), GetRows(), GetCValues().data());
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
}