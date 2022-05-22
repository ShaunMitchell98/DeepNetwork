#pragma once

#include "Activation.h"

namespace PyNet::Infrastructure::Activations {

	class Relu : public Activation {
	public:

		static auto factory() {
			return new Relu();
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) {

			Input = input;
			return Input->Max(0);
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {
			auto derivative = Input->Step();
			return dLoss_dOutput ^ *derivative;
		}
	};
}