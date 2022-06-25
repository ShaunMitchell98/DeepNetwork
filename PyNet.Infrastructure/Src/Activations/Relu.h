#pragma once

#include "Activation.h"

namespace PyNet::Infrastructure::Activations {

	class Relu : public Activation {
	public:

		static auto factory(unique_ptr<Matrix> input) {
			return new Relu(move(input));
		}

		Relu(unique_ptr<Matrix> input) : Activation(move(input)) {}

		void Initialise(size_t rows, size_t cols)
		{
			Input->Initialise(rows, cols, false);
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override {

			Input = input;
			return Input->Max(0);
		}

		unique_ptr<Matrix> Derivative(const Matrix& dLoss_dOutput) const override {
			return Input->Step();
		}
	};
}