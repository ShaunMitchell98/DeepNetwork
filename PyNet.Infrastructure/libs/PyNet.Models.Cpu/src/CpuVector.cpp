#include "CpuVector.h"

namespace PyNet::Models::Cpu {

	CpuVector::CpuVector(std::shared_ptr<Activation> activation) : Vector(activation) {}

	CpuVector::CpuVector(const CpuVector& v) : Vector(v._activation) {}

	std::unique_ptr<Vector> CpuVector::operator^(const Vector& v) {

		auto c = std::unique_ptr<Vector>(new CpuVector(this->_activation));
		c->Initialise(v.GetRows(), false);

		for (auto i = 0; i < v.GetRows(); i++) {
			c->SetValue(i, this->GetValue(i) * v.GetValue(i));
		}

		return std::move(c);
	}

	std::unique_ptr<Vector> CpuVector::CalculateActivationDerivative() {
		auto derivative = _activation->CalculateDerivative(*this);
		return std::unique_ptr<Vector>(new CpuVector(std::move(*derivative)));
	}


	std::unique_ptr<Vector> CpuVector::operator/(const double d) {
		auto result = Matrix::operator/(d);
		return std::unique_ptr<Vector>(new CpuVector(std::move(*result)));
	}
}
