#include "NetworkTrainer.h"
#include <ranges>

using namespace std::views;

namespace PyNet::Infrastructure 
{    
    void NetworkTrainer::TrainNetwork(double** inputLayers, double** expectedOutputs, int numberOfExamples, double baseLearningRate) 
    {
        auto batchNumber = 1;
        auto totalLossForCurrentEpoch = 0.0;
        auto learningRate = baseLearningRate;
        shared_ptr<Matrix> actualMatrix;

        auto tempTrainableLayers = all(_pyNetwork->Layers)
            | filter([](auto& layer) { return dynamic_cast<TrainableLayer*>(layer.get()); })
            | transform([](auto& layer) { return dynamic_cast<TrainableLayer*>(layer.get()); });

        auto trainableLayers = vector<TrainableLayer*>(tempTrainableLayers.begin(), tempTrainableLayers.end());

        auto expectedMatrix = _context->GetUnique<Matrix>();

        try
        {
            for (auto epoch = 0; epoch < _settings->Epochs; epoch++) 
            {
                for (auto i = 0; i < numberOfExamples; i++) 
                {
                    expectedMatrix->Set(_pyNetwork->Layers.back()->GetRows(), _pyNetwork->Layers.back()->GetCols(), expectedOutputs[i]);
                    TrainExamples(inputLayers[i], *expectedMatrix, batchNumber, learningRate, totalLossForCurrentEpoch, trainableLayers);
                }

                if (_vlSettings != nullptr) 
                {
                    VariableLearning(inputLayers, expectedOutputs, *actualMatrix, numberOfExamples, *expectedMatrix, totalLossForCurrentEpoch, learningRate,
                        trainableLayers);
                }

                totalLossForCurrentEpoch = 0.0;
            }
        }
        catch (const char* message)
        {
            _logger->LogLine(message);
            printf(message);
        }
	}

    void NetworkTrainer::TrainExamples(double* inputLayer, const Matrix& expectedOutput, int& batchNumber, double& learningRate, double& totalLossForCurrentEpoch,
        vector<TrainableLayer*> trainableLayers) 
    {
        auto actualMatrix = _networkRunner->Run(inputLayer);

        auto loss = _loss->CalculateLoss(expectedOutput, *actualMatrix);

        totalLossForCurrentEpoch += loss;

        auto lossDerivative = _loss->CalculateDerivative(expectedOutput, *actualMatrix);

        _backPropagator->Propagate(*_pyNetwork, *lossDerivative);

        if (batchNumber == _settings->BatchSize)
        {
            UpdateNetwork(learningRate, batchNumber, trainableLayers);
        }
        else
        {
            batchNumber++;
            _settings->NewBatch = false;
        }
    }

    void NetworkTrainer::UpdateNetwork(double learningRate, int& batchNumber, vector<TrainableLayer*> trainableLayers) 
    {
        _trainingAlgorithm->UpdateWeights(trainableLayers, learningRate, false);

        batchNumber = 1;
        _settings->NewBatch = true;
    }

    void NetworkTrainer::SetVLSettings(double errorThreshold, double lrDecrease, double lrIncrease) 
    {
        _vlSettings = make_unique<VariableLearningSettings>();
        _vlSettings->ErrorThreshold = errorThreshold;
        _vlSettings->LRDecrease = lrDecrease;
        _vlSettings->LRIncrease = lrIncrease;
	}

    void NetworkTrainer::VariableLearning(double** inputLayers, double** expectedOutputs, Matrix& actualMatrix, int numberOfExamples, Matrix& expectedMatrix,
        double totalLossForCurrentEpoch, double& learningRate, vector<TrainableLayer*>& trainableLayers) 
    {
        auto adjustedTotalLossForCurrentEpoch = 0.0;
        for (auto j = 0; j < numberOfExamples; j++)
        {
            _networkRunner->Run(inputLayers[j]);
            expectedMatrix.Set(actualMatrix.GetRows(), actualMatrix.GetCols(), expectedOutputs[j]);
            adjustedTotalLossForCurrentEpoch += _loss->CalculateLoss(expectedMatrix, actualMatrix);
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