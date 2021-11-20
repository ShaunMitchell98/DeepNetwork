#pragma once

#include "PyNet.Models/Activation.h"

namespace PyNet::Models::Cpu {

	class CpuLogistic : public Activation {

	public:

		static auto factory() {
			return new CpuLogistic{};
		}

		typedef Activation base;

		void Apply(PyNet::Models::Matrix& input);
		void CalculateDerivative(PyNet::Models::Matrix& input, PyNet::Models::Matrix& output);
	};
}
