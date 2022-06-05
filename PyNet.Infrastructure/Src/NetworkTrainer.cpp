#include "NetworkTrainer.h"
#include <iostream>

namespace PyNet::Infrastructure 
{    
    void NetworkTrainer::TrainNetwork(vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>> trainingPairs) 
    {
        _learningRate = _settings->BaseLearningRate;

        auto trainableLayers = _pyNetwork->GetTrainableLayers();

        auto exampleNumber = 1;

        for (auto epoch = 1; epoch <= _settings->Epochs; epoch++)
        {
            for (auto& trainingPair : trainingPairs)
            {
                TrainOnExample(trainingPair.first, *trainingPair.second, trainableLayers);
                cout << "Epoch: " << epoch << ", Example Number: " << exampleNumber << endl;
                exampleNumber++;
            }

            exampleNumber = 1;

            _vlService->RunVariableLearning(trainingPairs, _learningRate, _totalLossForCurrentEpoch);

            _totalLossForCurrentEpoch = 0.0;
        }
	}

    void NetworkTrainer::TrainOnExample(shared_ptr<Matrix> input, const Matrix& expectedOutput, vector<TrainableLayer*> trainableLayers) 
    {
        auto actualOutput = _networkRunner->Run(input);

        auto loss = _loss->CalculateLoss(expectedOutput, *actualOutput);

        _totalLossForCurrentEpoch += loss;

        auto lossDerivative = _loss->CalculateDerivative(expectedOutput, *actualOutput);

        _backPropagator->Propagate(*_pyNetwork, *lossDerivative);

        if (_batchNumber == _settings->BatchSize)
        {
            _trainingAlgorithm->UpdateWeights(trainableLayers, _learningRate, false);
            _batchNumber = 1;
            _settings->NewBatch = true;
        }
        else
        {
            _batchNumber++;
            _settings->NewBatch = false;
        }
    }
}