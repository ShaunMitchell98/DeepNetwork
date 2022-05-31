#pragma once

#include "Layer.h"
#include <memory>

using namespace std;

namespace PyNet::Infrastructure::Layers {

	class InputLayer : public Layer {
	private:

		InputLayer(unique_ptr<Matrix> input) : Layer(move(input)) {}

	public:

		static auto factory(unique_ptr<Matrix> input) {
			return new InputLayer(move(input));
		}

		void SetInput(double* input) {
			Input->Set(Input->GetRows(), Input->GetCols(), input);
		}

		void Initialise(size_t rows, size_t cols) {
			Input->Initialise(rows, cols, false);
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override {
			auto output = Input->Copy();
			output->Set(output->GetRows(), output->GetCols(), Input->GetAddress(1, 1));
			return output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {
			return dLoss_dOutput.Copy();
		}
	};
}