#pragma once

#include "Layer.h"
#include <memory>
#include "PyNet.DI/Context.h"

using namespace std;
using namespace PyNet::DI;

namespace PyNet::Infrastructure::Layers {

	class InputLayer : public Layer {
	private:
		unique_ptr<Matrix> _input;

		InputLayer(shared_ptr<Context> context) {
			_input = context->GetUnique<Matrix>();
		}

	public:

		~InputLayer() {
			auto a = 5;
		}

		static auto factory(shared_ptr<Context> context) {
			return new InputLayer(context);
		}

		void SetInput(double* input) {
			_input->Set(_input->GetRows(), _input->GetCols(), input);
		}

		void Initialise(size_t rows, size_t cols) {
			_input->Initialise(rows, cols, false);
		}

		size_t GetRows() const override {
			return _input->GetRows();
		}
		
		size_t GetCols() const override {
			return _input->GetCols();
		}

		unique_ptr<Matrix> Apply(unique_ptr<Matrix> input) override {
			auto output = _input->Copy();
			output->Set(output->GetRows(), output->GetCols(), _input->GetAddress(0, 0));
			return output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const {
			throw "Should not be called!";
		}
	};
}