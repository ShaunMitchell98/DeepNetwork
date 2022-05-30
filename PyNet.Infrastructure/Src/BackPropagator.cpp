#include "BackPropagator.h"
#include "Layers/TrainableLayer.h"

namespace PyNet::Infrastructure {

	void BackPropagator::Propagate(const PyNetwork& pyNetwork, Matrix& lossDerivative) const
	{
		unique_ptr<Matrix> dLoss_dOutput_Ptr;
		auto& dLoss_dOutput = lossDerivative;

		for (auto it = pyNetwork.Layers.rbegin(); it != pyNetwork.Layers.rend(); ++it) {

			auto& layer = *it;

			auto trainableLayer = dynamic_cast<TrainableLayer*>(layer.get());

			if (trainableLayer != nullptr)
			{
				trainableLayer->UpdateAdjustments(dLoss_dOutput);
			}

			dLoss_dOutput_Ptr = layer->dLoss_dInput(dLoss_dOutput);
			dLoss_dOutput = *dLoss_dOutput_Ptr;
		}
	}
}
