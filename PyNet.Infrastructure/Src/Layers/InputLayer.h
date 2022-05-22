#pragma once

#include "Layer.h"
#include <memory>
#include "PyNet.DI/Context.h"

using namespace std;
using namespace PyNet::DI;

namespace PyNet::Infrastructure::Layers {

	class InputLayer : public Layer {
	private:

		InputLayer(shared_ptr<Context> context) {
			Input = context->GetUnique<Matrix>();
		}

	public:

		static auto factory(shared_ptr<Context> context) {
			return new InputLayer(context);
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

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const {
			return dLoss_dOutput.Copy();
		}
	};
}