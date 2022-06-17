#pragma once

#include "Layer.h"
#include "../Settings.h"
#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"
#include "BernoulliGenerator.h"

using namespace PyNet::DI;

namespace PyNet::Infrastructure::Layers {

	class DropoutLayer : public Layer {
	private:
		double _rate = 1;
		size_t _rows = 0;
		size_t _cols = 0;
		shared_ptr<Settings> _settings;
		shared_ptr<BernoulliGenerator> _bernoulliGenerator;

		DropoutLayer(shared_ptr<Settings> settings, shared_ptr<BernoulliGenerator> bernoulliGenerator, unique_ptr<Matrix> input, unique_ptr<Matrix> output) : _settings{ settings }, 
			_bernoulliGenerator{ bernoulliGenerator }, Layer(move(input), move(output)) {}

	public:

		static auto factory(shared_ptr<Settings> settings, shared_ptr<BernoulliGenerator> bernoulliGenerator, unique_ptr<Matrix> input,
			unique_ptr<Matrix> output) {
			return new DropoutLayer(settings, bernoulliGenerator, move(input), move(output));
		}

		void Initialise(double rate, size_t rows, size_t cols) {
			_rate = rate;
			_rows = rows;
			_cols = cols;
		}

		shared_ptr<Matrix> ApplyInternal(shared_ptr<Matrix> input) override {

			auto output = input->Copy();

			unique_ptr<Matrix> droppedMatrix;
			if (_settings->RunMode == RunMode::Training) {
				droppedMatrix = _bernoulliGenerator->GetBernoulliVector(*input, _rate);
			}
			else {
				droppedMatrix = input->Copy();

				for (auto& m : *droppedMatrix) {
					m = _rate;
				}
			}

			*output = *(*input ^ *droppedMatrix);
			return output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {

			auto dLoss_dInput = dLoss_dOutput.Copy();
			dLoss_dInput->Set(dLoss_dInput->GetRows(), dLoss_dInput->GetCols(), dLoss_dOutput.GetAddress(1, 1));
			return dLoss_dInput;
		}
	};
}