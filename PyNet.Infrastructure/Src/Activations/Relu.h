#pragma once

#include "Activation.h"

namespace PyNet::Infrastructure::Activations {

	class Relu : public Activation {
	private:
		unique_ptr<Matrix> _input;
	public:

		static auto factory() {
			return new Relu();
		}

		unique_ptr<Matrix> Apply(unique_ptr<Matrix> input) {

			_input.swap(input);
			return _input->Max(0);
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {
			auto derivative = _input->Step();
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