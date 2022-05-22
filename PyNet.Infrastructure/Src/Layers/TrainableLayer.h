#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"
#include "Layer.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure::Layers {

	class TrainableLayer : public Layer {
	public:
		virtual shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) = 0;
		virtual unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const = 0;
		virtual void UpdateAdjustments(const Matrix& dLoss_dOutput) = 0;
		virtual Matrix& GetdLoss_dWeightSum() const = 0;
		virtual double GetdLoss_dBiasSum() const = 0;
		virtual Matrix& GetWeights() = 0;
		virtual double& GetBias() = 0;
	};
}