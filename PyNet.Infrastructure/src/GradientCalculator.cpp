#include "GradientCalculator.h"
#include "Layers/TrainableLayer.h"

namespace PyNet::Infrastructure {

	void GradientCalculator::CalculateGradients(vector<Layer*> layers, Matrix& lossDerivative)
	{
		unique_ptr<Matrix> dLoss_dOutput_Ptr;
		auto& dLoss_dOutput = lossDerivative;

		for (auto it = layers.rbegin(); it != layers.rend(); ++it) {

			auto layer = *it;

			auto trainableLayer = dynamic_cast<TrainableLayer*>(layer);

			if (trainableLayer != nullptr)
			{
				trainableLayer->UpdateAdjustments(dLoss_dOutput);
			}

			dLoss_dOutput_Ptr = layer->dLoss_dInput(dLoss_dOutput);
			dLoss_dOutput = *dLoss_dOutput_Ptr;
		}
	}
}
