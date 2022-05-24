#pragma once

#include "Layer.h"

namespace PyNet::Infrastructure::Layers {
	class FlattenLayer : public Layer {
	public:

		static auto factory(unique_ptr<Matrix> input) {
			return new FlattenLayer(move(input));
		}

		FlattenLayer(unique_ptr<Matrix> input) : Layer(move(input)) {}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) {
			
			Input = input;
			Output = input->Copy();
			Output->Set(Output->GetRows() * Output->GetCols(), 1, Output->GetAddress(1, 1));
			return Output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {

			auto dLoss_dInput = Input->Copy();
			dLoss_dInput->Set(dLoss_dInput->GetRows(), dLoss_dInput->GetCols(), dLoss_dOutput.GetAddress(1, 1));
			return dLoss_dInput;
		}
	};
}