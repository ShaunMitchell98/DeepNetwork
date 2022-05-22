#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure::Layers {

	class Layer {
	protected:
		shared_ptr<Matrix> Input;
		shared_ptr<Matrix> Output;
	public:
		size_t GetRows() const { return Input->GetRows(); }
		size_t GetCols() const { return Input->GetCols(); }
		virtual shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) = 0;
		virtual unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const = 0;
	};
}