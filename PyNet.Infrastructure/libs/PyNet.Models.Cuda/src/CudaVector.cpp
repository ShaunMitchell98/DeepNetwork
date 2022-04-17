#include "CudaVector.h"
#include "Matrix_Operations.h"

namespace PyNet::Models::Cuda {

	unique_ptr<Vector> CudaVector::CalculateActivationDerivative() {
		auto derivative = _activation->CalculateDerivative(this->operator const Matrix & ());
		return std::move(unique_ptr<Vector>(new CudaVector(std::move(*derivative))));
	}

	unique_ptr<Vector> CudaVector::operator+(const Vector& v) const {
		auto c = unique_ptr<Vector>(new CudaVector(*this));
		c->Initialise(GetRows(), false);
		matrix_add((this->operator const Matrix & ()).GetCValues(), v.GetCValues(), c->Values, this->GetRows(), this->GetCols());
		return std::move(c);
	}

	unique_ptr<Vector> CudaVector::operator-(const Vector& v) const {
		auto c = unique_ptr<Vector>(new CudaVector(*this));
		c->Initialise(GetRows(), false);
		matrix_subtract((this->operator const Matrix & ()).GetCValues(), v.GetCValues(), c->Values, this->GetRows(), this->GetCols());
		return std::move(c);
	}

	unique_ptr<Vector> CudaVector::operator^(const Vector& v) const {
		auto c = unique_ptr<Vector>(new CudaVector(this->_activation));
		c->Initialise(v.GetRows(), false);

		for (auto i = 0; i < v.GetRows(); i++) {
			(*c)[i] = (*this)[i] * v[i];
		}

		return std::move(c);
	}

	unique_ptr<Vector> CudaVector::operator/(const double d) const {
		auto result = Matrix::operator/(d);
		return std::move(unique_ptr<Vector>(new CudaVector(std::move(*result))));
	}
}