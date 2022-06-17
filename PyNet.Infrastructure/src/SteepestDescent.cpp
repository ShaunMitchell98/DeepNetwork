#include "SteepestDescent.h"
#include "Layers/TrainableLayer.h"
#include <format>

namespace PyNet::Infrastructure {

    void SteepestDescent::UpdateWeights(vector<TrainableLayer*> layers, double learningRate, bool reverse) const 
    {
        for (auto i = 0; i < layers.size(); i++)
        {
            auto average_dLoss_dBias = layers[i]->DLoss_dBiasSum / _settings->BatchSize;
            auto biasAdjustment = average_dLoss_dBias * learningRate;

            auto average_dLoss_dWeight = *layers[i]->DLoss_dWeightSum / _settings->BatchSize;

            //_logger->LogLine("Average dLoss_dWeight: ");
            //_logger->LogMatrix(*average_dLoss_dWeight);

            auto weightAdjustment = *average_dLoss_dWeight * learningRate;

            if (reverse)
            {
                *layers[i]->Weights = *(*layers[i]->Weights + *weightAdjustment);
                layers[i]->Bias = layers[i]->Bias + biasAdjustment;
            }
            else
            {
                *layers[i]->Weights = *(*layers[i]->Weights - *weightAdjustment);

                //_logger->LogLine("New weights: ");
                //_logger->LogMatrix(*layers[i]->Weights);

                layers[i]->Bias = layers[i]->Bias - biasAdjustment;
            }

            _logger->LogLine("Bias for trainable layer {} is {}", make_format_args(i, layers[i]->Bias));
            auto weight = (*layers[i]->Weights)(1, 1);
            _logger->LogLine("Weights for trainable layer {} are {}", make_format_args(i, weight));
        }
    }
}