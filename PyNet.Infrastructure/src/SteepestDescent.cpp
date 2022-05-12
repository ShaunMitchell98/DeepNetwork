#include "SteepestDescent.h"
#include "Layers/TrainableLayer.h"
#include <ranges>

namespace PyNet::Infrastructure {

    void SteepestDescent::UpdateWeights(vector<TrainableLayer*> layers, double learningRate, bool reverse) const {

        auto castOp = [](Layer& layer) {
            return dynamic_cast<TrainableLayer*>(&layer);
        };

        //auto trainableLayers = views::all(layers)
        //    | views::filter([castOp](Layer& layer) {return castOp(layer); }) 
        //    | views::transform([castOp](Layer& layer) {return castOp(layer); }) 
        //    | views::reverse;

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

