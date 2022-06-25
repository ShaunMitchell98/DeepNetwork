#pragma once

#include "PyNet.Models/Relu.h"

namespace PyNet::Models::Cpu {

	class CpuRelu : public Relu {

	public:

		static auto factory() {
			return new CpuRelu();
		}

		void Apply(Matrix& input) override {

		}

		unique_ptr<Matrix> CalculateDerivative(const Matrix& input) override {

		}
	};
}