#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure::Layers {

	class Layer {
	public:
		virtual size_t GetRows() const = 0;
		virtual size_t GetCols() const = 0;
		virtual unique_ptr<Matrix> Apply(unique_ptr<Matrix> input) = 0;
		virtual unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const = 0;
	};
}