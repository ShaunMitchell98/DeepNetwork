#pragma once

#include "Layers/Layer.h"

using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure::Tests::Layers
{
	class FakeLayer : public Layer 
	{
		private:
		double _value = 0;
	public:
		FakeLayer(unique_ptr<Matrix> input) : Layer(move(input)) {}

		static auto factory(unique_ptr<Matrix> input) 
		{
			return new FakeLayer(move(input));
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
