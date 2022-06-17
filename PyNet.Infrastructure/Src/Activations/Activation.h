#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure::Activations {

	enum class ActivationFunctionType {
		Logistic,
		Relu
	};

	class Activation {

		protected:
		shared_ptr<Matrix> Input;
		shared_ptr<Matrix> Output;
		public:
		Activation(shared_ptr<Matrix> input) : Input(input) {}


		size_t GetRows() const { return Input->GetRows(); }
		size_t GetCols() const { return Input->GetCols(); }
		virtual shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) = 0;
		virtual unique_ptr<Matrix> Derivative(const Matrix& dLoss_dOutput) const = 0;
		virtual ~Activation() = default;
	};
}