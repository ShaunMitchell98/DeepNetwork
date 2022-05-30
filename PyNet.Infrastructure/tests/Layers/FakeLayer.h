#pragma once

#include "Layers/TrainableLayer.h"

using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure::Tests::Layers
{
	class FakeLayer : public TrainableLayer 
	{
		private:
		double _value = 0;
		
	public:
		FakeLayer(unique_ptr<Matrix> input, unique_ptr<Matrix> dLoss_dWeightSum, unique_ptr<Matrix> weights) : TrainableLayer(move(dLoss_dWeightSum),
			move(weights), move(input)) {}

		bool Adjusted = false;

		static auto factory(unique_ptr<Matrix> input, unique_ptr<Matrix> dLoss_dWeightSum, unique_ptr<Matrix> weights) 
		{
			return new FakeLayer(move(input), move(dLoss_dWeightSum), move(weights));
		}

		void SetValue(double value) 
		{
			_value = value;
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override 
		{
			auto output = shared_ptr<Matrix>(input->Copy().release());

			for (auto& element : *output) 
			{
				element = _value;
			}

			return output;
		}

		void UpdateAdjustments(const Matrix& dLoss_dOutput) override 
		{
			Adjusted = true;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const 
		{
			auto output = dLoss_dOutput.Copy();

			for (auto& element : *output) 
			{
				element = _value;
			}

			return output;
		}
	};
}
