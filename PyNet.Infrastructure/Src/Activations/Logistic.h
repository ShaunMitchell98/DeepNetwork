#pragma once

#include "Activation.h"
#include <memory>

namespace PyNet::Infrastructure::Activations {

	class Logistic : public Activation {

		Logistic(unique_ptr<Matrix> input) : Activation(move(input)) { }
	public:

		static auto factory(unique_ptr<Matrix> input) {
			return new Logistic(move(input));
		}

		void Initialise(size_t rows, size_t cols) {
			Input->Initialise(rows, cols, false);
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override {

			Input = input;
			return (*((-*Input)->Exp()) + 1)->Reciprocal();
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {
			auto derivative = *Input->Exp() * *((*(*Input->Exp() + 1) * *(*Input->Exp() + 1))->Reciprocal());
			return dLoss_dOutput ^ *derivative;
		}
	};
}