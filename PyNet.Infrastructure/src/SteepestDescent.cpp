#include "SteepestDescent.h"
#include "Layers/TrainableLayer.h"

namespace PyNet::Infrastructure {

    void SteepestDescent::UpdateWeights(vector<TrainableLayer*> layers, double learningRate, bool reverse) const {

        for (auto layer : layers)
        {
            auto average_dLoss_dBias = layer->GetdLoss_dBiasSum() / _settings->BatchSize;
            auto biasAdjustment = average_dLoss_dBias * learningRate;

            auto average_dLoss_dWeight = layer->GetdLoss_dWeightSum() / _settings->BatchSize;
            auto weightAdjustment = *average_dLoss_dWeight * learningRate;

            if (reverse) {
                
                layer->GetWeights() = *(layer->GetWeights() + *weightAdjustment);
                layer->GetBias() = layer->GetBias() + biasAdjustment;
            }
            else {
                layer->GetWeights() = *(layer->GetWeights() - *weightAdjustment);
                layer->GetBias() = layer->GetBias() - biasAdjustment;
            }
        }
    }
}

