#pragma once
#include <vector>
#include <memory>
#include "Layers/TrainableLayer.h"

using namespace std;
using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure {
	struct PyNetwork
	{
	public:
		vector<unique_ptr<Layer>> Layers = vector<unique_ptr<Layer>>();

		static auto factory() {
			return new PyNetwork();
		}

		vector<TrainableLayer*> GetTrainableLayers() 
		{
            auto trainableLayers = vector<TrainableLayer*>();

            for (auto& layer : Layers)
            {
                auto trainableLayer = dynamic_cast<TrainableLayer*>(layer.get());

                if (trainableLayer != nullptr)
                {
                    trainableLayers.push_back(trainableLayer);
                }
            }
            return trainableLayers;
		}
	};
}

