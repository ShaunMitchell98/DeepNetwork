#include "VLService.h"

namespace PyNet::Infrastructure 
{
	void VLService::RunVariableLearning(vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>> trainingPairs, double& learningRate, double totalLossForCurrentEpoch)
	{
        if (_vlSettings->LRDecrease == 0)
        {
            return;
        }

        auto trainableLayers = _pyNetwork->GetTrainableLayers();

        auto adjustedTotalLossForCurrentEpoch = 0.0;

        for (auto& trainingPair : trainingPairs) 
        {
            auto actualMatrix = _networkRunner->Run(trainingPair.first);
            adjustedTotalLossForCurrentEpoch += _loss->CalculateLoss(*trainingPair.second, *actualMatrix);
        }

        if (adjustedTotalLossForCurrentEpoch > (1 + _vlSettings->ErrorThreshold) * totalLossForCurrentEpoch)
        {
            learningRate = learningRate * _vlSettings->LRDecrease;
            _trainingAlgorithm->UpdateWeights(trainableLayers, learningRate, true);
            _settings->Momentum = 0;
        }
        else if (adjustedTotalLossForCurrentEpoch > totalLossForCurrentEpoch)
        {
            _settings->Momentum = _settings->Momentum;
        }
        else
        {
            learningRate = learningRate * _vlSettings->LRIncrease;
            _settings->Momentum = _settings->Momentum;
        }
    }
}