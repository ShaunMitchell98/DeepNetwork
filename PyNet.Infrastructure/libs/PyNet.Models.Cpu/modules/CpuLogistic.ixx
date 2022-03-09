module;
#include <memory>
export module PyNet.Models.Cpu:CpuLogistic;

using namespace std;

import PyNet.Models;
import :CpuMatrix;

export namespace PyNet::Models::Cpu {

	class CpuLogistic : public Activation {

	public:

		static auto factory() {
			return new CpuLogistic{};
		}

		typedef Activation base;

		void Apply(Matrix& input) {

			for (auto i = 0; i < input.GetRows(); i++) {
				for (auto j = 0; j < input.GetCols(); j++) {

					input(i, j) = 1 / (1 + exp(-input(i, j)));
				}
			}
		}

		unique_ptr<Matrix> CalculateDerivative(const Matrix& input) {

			auto output = make_unique<CpuMatrix>();
			output->Initialise(input.GetRows(), input.GetCols(), false);

			for (auto i = 0; i < input.GetRows(); i++) {
				for (auto j = 0; j < input.GetCols(); j++) {
					(*output)(i, j) = exp(input(i, j)) / ((1 + exp(input(i, j)) * (1 + exp(input(i, j)))));
				}
			}

			return move(output);
		}
	};
}
