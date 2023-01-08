#pragma once

#include "Activation.h"
#include <memory>

namespace PyNet::Infrastructure::Activations {

	class Logistic : public Activation {

		private:
		shared_ptr<ILogger> _logger;

		Logistic(unique_ptr<Matrix> input, shared_ptr<ILogger> logger) : _logger(logger), Activation(move(input)) { }
	public:

		static auto factory(unique_ptr<Matrix> input, shared_ptr<ILogger> logger) {
			return new Logistic(move(input), logger);
		}

		void Initialise(size_t rows, size_t cols) {
			Input->Initialise(rows, cols, false);
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override {

			_logger->LogDebug("Logistic Input is: ");
			_logger->LogDebugMatrix(*input);

			Output = input;

			for (int i = 1; i <= input->GetRows(); i++)
			{
				for (auto j = 1; j <= input->GetCols(); j++)
				{
					(*Output)(i, j) = 1 / (1 + exp(-(*input)(i, j)));
				}
			}

			_logger->LogDebug("Logistic Output is: ");
			_logger->LogDebugMatrix(*Output);

			return Output;
		}

		unique_ptr<Matrix> Derivative(const Matrix& input) const override {
			//auto derivative = *Input->Exp() * *((*(*Input->Exp() + 1) * *(*Input->Exp() + 1))->Reciprocal());
			//return dLoss_dOutput ^ *derivative;

			auto derivative = input.Copy();

			for (auto i = 1; i <= input.GetRows(); i++)
			{
				for (auto j = 1; j <= input.GetCols(); j++)
				{
					(*derivative)(i, j) = exp(input(i, j)) / ((1 + exp(input(i, j)) * (1 + exp(input(i, j)))));
				}
			}

			//_logger->LogLine("Logistic Derivative is :");
			//_logger->LogMatrix(*derivative);

			//auto value = dLoss_dOutput ^ *derivative;

		/*	_logger->LogLine("Logistic dLoss_dInput is: ");
			_logger->LogMatrix(*value);*/
			return derivative;
		}
	};
}