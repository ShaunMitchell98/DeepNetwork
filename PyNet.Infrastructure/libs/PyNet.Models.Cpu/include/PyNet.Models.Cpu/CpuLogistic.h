#pragma once
#include <math.h>
#include "PyNet.Models/Logistic.h"

using namespace std;

namespace PyNet::Models::Cpu {

	class CpuLogistic : public Logistic {

		shared_ptr<PyNet::DI::Context> _context;

	public:

		static auto factory(shared_ptr<PyNet::DI::Context> context) {
			return new CpuLogistic{context};
		}

		CpuLogistic(shared_ptr<PyNet::DI::Context> context) : _context{ context } {}

		typedef Activation base;

		void Apply(Matrix& input) override {

			for (int i = 0; i < input.GetRows(); i++) {
				for (auto j = 0; j < input.GetCols(); j++) {

					input(i, j) = 1 / (1 + exp(-input(i, j)));
				}
			}
		}

		unique_ptr<Matrix> CalculateDerivative(const Matrix& input) override {

			auto output = _context->GetUnique<Matrix>();
			output->Initialise(input.GetRows(), input.GetCols(), false);

			for (auto i = 0; i < input.GetRows(); i++) {
				for (auto j = 0; j < input.GetCols(); j++) {
					(*output)(i, j) = exp(input(i, j)) / ((1 + exp(input(i, j)) * (1 + exp(input(i, j)))));
				}
			}

			return move(output);
		}

		~CpuLogistic() override = default;
	};
}
