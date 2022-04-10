#include "CpuVector.h"

namespace PyNet::Models::Cpu {

	CpuVector::CpuVector(std::shared_ptr<Activation> activation) : Vector(activation) {}

	CpuVector::CpuVector(const CpuVector& v) : Vector(v._activation) {}

	unique_ptr<Vector> CpuVector::operator^(const Vector& v) const {

		auto c = unique_ptr<Vector>(new CpuVector(this->_activation));
		c->Initialise(v.GetRows(), false);

		for (auto i = 0; i < v.GetRows(); i++) {
			(*c)[i] = (*this)[i] * v[i];
		}

		return move(c);
	}

	unique_ptr<Vector> CpuVector::CalculateActivationDerivative() const {
		auto derivative = _activation->CalculateDerivative(*this);
		return move(unique_ptr<Vector>(new CpuVector(move(*derivative))));
	}


	unique_ptr<Vector> CpuVector::operator/(const double d) const {
		auto result = Matrix::operator/(d);
		return move(unique_ptr<Vector>(new CpuVector(move(*result))));
	}
}
