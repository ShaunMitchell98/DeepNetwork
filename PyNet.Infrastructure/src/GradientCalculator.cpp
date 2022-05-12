#include "GradientCalculator.h"
#include <ranges>
#include "Layers/TrainableLayer.h"

using namespace std::ranges::views;

namespace PyNet::Infrastructure {

	void GradientCalculator::CalculateGradients(vector<TrainableLayer*> layers, Matrix& lossDerivative)
	{
		auto castOp = [](Layer& layer) {
			return dynamic_cast<TrainableLayer*>(&layer);
		};

		unique_ptr<Matrix> dLoss_dOutput_Ptr;
		auto& dLoss_dOutput = lossDerivative;

		for (auto layer : layers) {
			layer->UpdateAdjustments(dLoss_dOutput);
			dLoss_dOutput_Ptr = layer->dLoss_dInput(dLoss_dOutput);
			dLoss_dOutput = *dLoss_dOutput_Ptr;
		}
	}
}
