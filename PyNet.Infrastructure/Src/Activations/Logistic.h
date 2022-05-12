#pragma once

#include "Activation.h"
#include <memory>

namespace PyNet::Infrastructure::Activations {

	class Logistic : public Activation {
	private:
		unique_ptr<Matrix> _input;
	public:

		static auto factory() {
			return new Logistic();
		}

		unique_ptr<Matrix> Apply(unique_ptr<Matrix> input) override {

			_input.swap(input);
			return (*((-*_input)->Exp()) + 1)->Reciprocal();
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {
			auto derivative = *_input->Exp() * *((*(*_input->Exp() + 1) * *(*_input->Exp() + 1))->Reciprocal());
			return dLoss_dOutput ^ *derivative;
		}

		size_t GetRows() const override {
			return _input->GetRows();
		}

		size_t GetCols() const override {
			return _input->GetCols();
		}
	};
}