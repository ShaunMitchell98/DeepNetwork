#include "NetworkTrainer.h"
#include <iostream>

namespace PyNet::Infrastructure 
{    
    void NetworkTrainer::TrainNetwork(vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>> trainingPairs) 
    {
        _learningRate = _settings->BaseLearningRate;

        auto trainableLayers = _pyNetwork->GetTrainableLayers();

        _trainingState->ExampleNumber = 1;

        for (_trainingState->Epoch = 1; _trainingState->Epoch <= _settings->Epochs; _trainingState->Epoch++)
        {
            for (auto& trainingPair : trainingPairs)
            {
                TrainOnExample(trainingPair.first, *trainingPair.second, trainableLayers);
                cout << "Epoch: " << _trainingState->Epoch << ", Example Number: " << _settings->StartExampleNumber + _trainingState->ExampleNumber << endl;
                _trainingState->ExampleNumber++;
            }

            _trainingState->ExampleNumber = 1;

            //_vlService->RunVariableLearning(trainingPairs, _learningRate, _totalLossForCurrentEpoch);

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
            _trainingState->NewBatch = true;
        }
        else
        {
            _batchNumber++;
            _trainingState->NewBatch = false;
        }
    }
}