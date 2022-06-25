#pragma once

#include "Layer.h"

namespace PyNet::Infrastructure::Layers {
	class FlattenLayer : public Layer {
	public:

		static auto factory(unique_ptr<Matrix> input, unique_ptr<Matrix> output) {
			return new FlattenLayer(move(input), move(output));
		}

		FlattenLayer(unique_ptr<Matrix> input, unique_ptr<Matrix> output) : Layer(move(input), move(output)) {}

		void Initialise(size_t rows, size_t cols)
		{
			Input->Initialise(rows, cols, false);
			Output->Initialise(rows * cols, 1, false);
		}

		shared_ptr<Matrix> ApplyInternal(shared_ptr<Matrix> input) override {
			
			Input = input;
			Output = input->Copy();
			Output->Set(Output->GetRows() * Output->GetCols(), 1, input->GetAddress(1, 1));
			return Output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {

			auto dLoss_dInput = Input->Copy();
			dLoss_dInput->Set(dLoss_dInput->GetRows(), dLoss_dInput->GetCols(), dLoss_dOutput.GetAddress(1, 1));
			return dLoss_dInput;
		}
	};
}