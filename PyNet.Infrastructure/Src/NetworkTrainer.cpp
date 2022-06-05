#include "NetworkTrainer.h"
#include <iostream>

namespace PyNet::Infrastructure 
{    
    void NetworkTrainer::TrainNetwork(vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>> trainingPairs) 
    {
        auto batchNumber = 1;
        auto totalLossForCurrentEpoch = 0.0;
        auto learningRate = _settings->BaseLearningRate;

        auto trainableLayers = _pyNetwork->GetTrainableLayers();

        auto iteration = 1;

        for (auto epoch = 0; epoch < _settings->Epochs; epoch++)
        {
            for (auto& trainingPair : trainingPairs)
            {
                TrainOnExample(trainingPair.first, *trainingPair.second, batchNumber, learningRate, totalLossForCurrentEpoch, trainableLayers);
                cout << "Iteration: " << iteration << endl;
                iteration ++;
            }

            iteration = 1;

            _vlService->RunVariableLearning(trainingPairs, learningRate, totalLossForCurrentEpoch);

            totalLossForCurrentEpoch = 0.0;
        }
	}

    void NetworkTrainer::TrainOnExample(shared_ptr<Matrix> input, const Matrix& expectedOutput, int& batchNumber, double& learningRate, double& totalLossForCurrentEpoch,
        vector<TrainableLayer*> trainableLayers) 
    {
        auto actualOutput = _networkRunner->Run(input);

        auto loss = _loss->CalculateLoss(expectedOutput, *actualOutput);

        totalLossForCurrentEpoch += loss;

        auto lossDerivative = _loss->CalculateDerivative(expectedOutput, *actualOutput);

        _backPropagator->Propagate(*_pyNetwork, *lossDerivative);

        if (batchNumber == _settings->BatchSize)
        {
            _trainingAlgorithm->UpdateWeights(trainableLayers, learningRate, false);
            batchNumber = 1;
            _settings->NewBatch = true;
        }
        else
        {
            batchNumber++;
            _settings->NewBatch = false;
        }
    }
}