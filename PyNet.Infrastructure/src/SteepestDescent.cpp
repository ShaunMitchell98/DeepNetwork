#include "SteepestDescent.h"
#include "Layers/TrainableLayer.h"

namespace PyNet::Infrastructure {

    void SteepestDescent::UpdateWeights(vector<TrainableLayer*> layers, double learningRate, bool reverse) const {

        for (auto layer : layers)
        {
            auto average_dLoss_dBias = layer->DLoss_dBiasSum / _settings->BatchSize;
            auto biasAdjustment = average_dLoss_dBias * learningRate;

            auto average_dLoss_dWeight = *layer->DLoss_dWeightSum / _settings->BatchSize;
            auto weightAdjustment = *average_dLoss_dWeight * learningRate;

            if (reverse) {
                
                *layer->Weights = *(*layer->Weights + *weightAdjustment);
                layer->Bias = layer->Bias + biasAdjustment;
            }
            else {
                *layer->Weights = *(*layer->Weights - *weightAdjustment);
                layer->Bias = layer->Bias - biasAdjustment;
            }
        }
    }
}