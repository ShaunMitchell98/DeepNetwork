#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"
#include "Layer.h"

using namespace PyNet::Models;

namespace PyNet::Infrastructure::Layers {

	class TrainableLayer : public Layer {
	public:
		double DLoss_dBiasSum;
		unique_ptr<Matrix> DLoss_dWeightSum;
		unique_ptr<Matrix> Weights;
		double Bias;

		TrainableLayer(unique_ptr<Matrix> dLoss_dWeightSum, unique_ptr<Matrix> weights, shared_ptr<Matrix> input)
			: DLoss_dWeightSum{ move(dLoss_dWeightSum) }, Weights{ move(weights) }, DLoss_dBiasSum{ 0.01 }, Layer(input) 
		{
			Bias = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX) * 1000);
		}

		virtual shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) = 0;
		virtual unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const = 0;
		virtual void UpdateAdjustments(const Matrix& dLoss_dOutput) = 0;
	};
}