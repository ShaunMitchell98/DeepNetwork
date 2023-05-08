#pragma once

#include <memory>
#include "../Activations/Activation.h"
#include "XmlWriter.h"
#include "XmlReader.h"
#include "PyNet.Models/Matrix.h"
#include <type_traits>

using namespace PyNet::Models;
using namespace PyNet::Infrastructure;
using namespace PyNet::Infrastructure::Activations;

namespace PyNet::Infrastructure::Layers {

	class Layer {
	protected:
		shared_ptr<Matrix> Input;
		shared_ptr<Matrix> Output;
		unique_ptr<Activation> _activation;
	public:

		double DropoutRate = 1.0;

		Layer(shared_ptr<Matrix> input, shared_ptr<Matrix> output) : Input{input }, Output(output) {}

		size_t GetRows() const { return Output->GetRows(); }
		size_t GetCols() const { return Output->GetCols(); }

		virtual shared_ptr<Matrix> ApplyInternal(shared_ptr<Matrix> input) = 0;
		virtual unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const = 0;
		virtual void Serialize(XmlWriter& writer) const = 0;
		virtual void Deserialize(XmlReader& reader) = 0;
		virtual ~Layer() = default;

		void SetActivation(unique_ptr<Activation> activation)
		{
			_activation = move(activation);
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input)
		{
			auto output = ApplyInternal(input);

			if (_activation.get() != nullptr) 
			{
				output = _activation->Apply(output);
			}

			return output;
		}

		shared_ptr<Matrix> ActivationDerivative() {
			return _activation->Derivative(*Output);
		}
	};
}