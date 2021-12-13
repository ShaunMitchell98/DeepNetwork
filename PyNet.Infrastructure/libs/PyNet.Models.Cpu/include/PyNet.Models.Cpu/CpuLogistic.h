#pragma once

#include "PyNet.Models/Activation.h"

namespace PyNet::Models::Cpu {

	class CpuLogistic : public Activation {

		std::shared_ptr<PyNet::DI::Context> _context;

	public:

		static auto factory(std::shared_ptr<PyNet::DI::Context> context) {
			return new CpuLogistic{context};
		}

		CpuLogistic(std::shared_ptr<PyNet::DI::Context> context) : _context{ context } {}

		typedef Activation base;

		void Apply(Matrix& input) {

			for (int i = 0; i < input.GetRows(); i++) {
				for (auto j = 0; j < input.GetCols(); j++) {

					input.SetValue(i, j, 1 / (1 + exp(-input.GetValue(i, j))));
				}
			}
		}

		std::unique_ptr<Matrix> CalculateDerivative(Matrix& input) {

			auto output = _context->GetUnique<Matrix>();
			output->Initialise(input.GetRows(), input.GetCols(), false);

			for (auto i = 0; i < input.GetRows(); i++) {
				for (auto j = 0; j < input.GetCols(); j++) {
					output->SetValue(i, j, exp(input.GetValue(i, j)) / ((1 + exp(input.GetValue(i, j)) * (1 + exp(input.GetValue(i, j))))));
				}
			}

			return std::move(output);
		}
	};
}
