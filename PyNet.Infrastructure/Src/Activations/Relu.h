#pragma once

#include "Activation.h"

namespace PyNet::Infrastructure::Activations {

	class Relu : public Activation {
	public:

		static auto factory(unique_ptr<Matrix> input) {
			return new Relu(move(input));
		}

		Relu(unique_ptr<Matrix> input) : Activation(move(input)) {}

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