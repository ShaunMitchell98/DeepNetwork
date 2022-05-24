#pragma once

#include "PyNet.Models/Matrix.h"
#include "TrainableLayer.h"
#include "AdjustmentCalculator.h"
#include <numeric>
#include <memory>

using namespace std;

namespace PyNet::Infrastructure::Layers
{
	class DenseLayer : public TrainableLayer 
	{
	private:
		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;

		unique_ptr<Matrix> dLoss_dWeight(const Matrix& inputLayer, const Matrix& dLoss_dOutput) const 
		{
			return dLoss_dOutput * *~inputLayer;
		}

		double dLoss_dBias(const Matrix& dLoss_dOutput) const 
		{
			return accumulate(dLoss_dOutput.begin(), dLoss_dOutput.end(), 0.0);
		}

	public:

		DenseLayer(shared_ptr<AdjustmentCalculator> adjustmentCalculator, unique_ptr<Matrix> weights, unique_ptr<Matrix> dLoss_dWeightSum,
			unique_ptr<Matrix> input) : _adjustmentCalculator{ adjustmentCalculator }, TrainableLayer(move(dLoss_dWeightSum), move(weights), move(input)) {}

		static auto factory(shared_ptr<AdjustmentCalculator> adjustmentCalculator, unique_ptr<Matrix> weights, unique_ptr<Matrix> dLoss_dWeightSum,
		unique_ptr<Matrix> input) 
		{
			return new DenseLayer(adjustmentCalculator, move(weights), move(dLoss_dWeightSum), move(input));
		}

		void Initialise(size_t rows, size_t cols)
		{
			Weights->Initialise(rows, cols, true);
			DLoss_dBiasSum = 0;
			DLoss_dWeightSum->Initialise(rows, cols, false);
			Input->Initialise(cols, 1, false);
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override 
		{

			Input = input;
			Output = *Weights * *Input;

			*Output = *(* Output + Bias);

			return Output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override 
		{
			return *~*Weights * dLoss_dOutput;
		}

		void UpdateAdjustments(const Matrix& dLoss_dOutput) override
		{
			DLoss_dBiasSum = _adjustmentCalculator->CalculateBiasAdjustment(dLoss_dBias(dLoss_dOutput), DLoss_dBiasSum);
			_adjustmentCalculator->CalculateWeightAdjustment(*dLoss_dWeight(*Input, dLoss_dOutput), *DLoss_dWeightSum);
		}
	};
}