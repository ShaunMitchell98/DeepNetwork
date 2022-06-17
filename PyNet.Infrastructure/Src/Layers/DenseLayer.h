#pragma once

#include "PyNet.Models/Matrix.h"
#include "TrainableLayer.h"
#include "AdjustmentCalculator.h"
#include "PyNet.Models/ILogger.h"
#include <numeric>
#include <memory>

using namespace std;

namespace PyNet::Infrastructure::Layers
{
	class DenseLayer : public TrainableLayer 
	{
	private:
		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
		shared_ptr<ILogger> _logger;

		unique_ptr<Matrix> dLoss_dWeight(const Matrix& dLoss_dOutput) const 
		{
			return dLoss_dOutput * *~*Input;
		}

		double dLoss_dBias(const Matrix& dLoss_dOutput) const 
		{
			return accumulate(dLoss_dOutput.begin(), dLoss_dOutput.end(), 0.0);
		}

	public:

		DenseLayer(shared_ptr<AdjustmentCalculator> adjustmentCalculator, unique_ptr<Matrix> weights, unique_ptr<Matrix> dLoss_dWeightSum,
			unique_ptr<Matrix> input, unique_ptr<Matrix> output, shared_ptr<ILogger> logger) : _adjustmentCalculator{ adjustmentCalculator }, 
			TrainableLayer(move(dLoss_dWeightSum), move(weights), move(input), move(output)), _logger(logger){}

		static auto factory(shared_ptr<AdjustmentCalculator> adjustmentCalculator, unique_ptr<Matrix> weights, unique_ptr<Matrix> dLoss_dWeightSum,
		unique_ptr<Matrix> input, unique_ptr<Matrix> output, shared_ptr<ILogger> logger) 
		{
			return new DenseLayer(adjustmentCalculator, move(weights), move(dLoss_dWeightSum), move(input), move(output), logger);
		}

		void Initialise(size_t rows, size_t cols)
		{
			Weights->Initialise(rows, cols, true);
			DLoss_dBiasSum = 0;
			DLoss_dWeightSum->Initialise(rows, cols, false);
			Input->Initialise(cols, 1, false);
			Output->Initialise(rows, 1, false);
		}

		shared_ptr<Matrix> ApplyInternal(shared_ptr<Matrix> input) override 
		{
			//_logger->LogLine("Dense Layer Input is: ");
			//_logger->LogMatrix(*input);

			Input = input;
			Output = *Weights * *Input;

			//_logger->LogLine("Weight * Input: ");
			//_logger->LogMatrix(*Output);

			//*Output = *(* Output + Bias);

			//_logger->LogLine("Output + Bias: ");
			//_logger->LogMatrix(*Output);

			return Output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override 
		{
			auto dLoss_dInput = *~*Weights * dLoss_dOutput;

			if (_activation.get() != nullptr)
			{
				dLoss_dInput = *dLoss_dInput ^ *_activation->Derivative(*Input);
			}

			//_logger->LogLine("dLoss_dInput is: ");
			//_logger->LogMatrix(*dLoss_dInput);

			return dLoss_dInput;
		}

		void UpdateAdjustments(const Matrix& dLoss_dOutput) override
		{
			auto dLoss_dWeightValue = dLoss_dWeight(dLoss_dOutput);

			//_logger->LogLine("dLoss_dWeight: ");
			//_logger->LogMatrix(*dLoss_dWeightValue);

			DLoss_dBiasSum = _adjustmentCalculator->CalculateBiasAdjustment(dLoss_dBias(dLoss_dOutput), DLoss_dBiasSum);
			_adjustmentCalculator->CalculateWeightAdjustment(*dLoss_dWeightValue, *DLoss_dWeightSum);
		}
	};
}