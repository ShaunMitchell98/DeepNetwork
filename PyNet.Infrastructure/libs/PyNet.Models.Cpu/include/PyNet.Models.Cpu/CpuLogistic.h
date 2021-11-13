#pragma once

#include "PyNet.Models/Activation.h"

namespace PyNet::Models::Cpu {

	class CpuLogistic : public Activation {
	public:
		void Apply(PyNet::Models::Matrix& input);
		void CalculateDerivative(PyNet::Models::Matrix& input, PyNet::Models::Matrix& output);
	};
}
