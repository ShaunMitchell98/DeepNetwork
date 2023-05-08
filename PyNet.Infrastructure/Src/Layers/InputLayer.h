#pragma once

#include "Layer.h"
#include <memory>

using namespace std;

namespace PyNet::Infrastructure::Layers 
{
	class InputLayer : public Layer 
	{
	private:

	InputLayer(unique_ptr<Matrix> input, unique_ptr<Matrix> output) : Layer(move(input), move(output)) {}

		public:

		static auto factory(unique_ptr<Matrix> input, unique_ptr<Matrix> output)
		{
			return new InputLayer(move(input), move(output));
		}

		void SetInput(shared_ptr<Matrix> input) 
		{
			Input = input;
		}

		void Initialise(size_t rows, size_t cols)
		{
			Input->Initialise(rows, cols);
			Output->Initialise(rows, cols);
		}

		shared_ptr<Matrix> ApplyInternal(shared_ptr<Matrix> input) override 
		{
			auto output = Input->Copy();
			output->Set(output->GetRows(), output->GetCols(), Input->GetAddress(1, 1));
			return output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override
		{
			return dLoss_dOutput.Copy();
		}

		void Serialize(XmlWriter& writer) const 
		{
			writer.StartElement("Type");
			writer.WriteString(typeid(InputLayer).name());
			writer.EndElement();
		}
	};
}