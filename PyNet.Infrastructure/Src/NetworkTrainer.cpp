#include "NetworkTrainer.h"
#include <ranges>

using namespace std::views;

namespace PyNet::Infrastructure {
    
    void NetworkTrainer::TrainNetwork(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate, double momentum, int epochs) {
        auto batchNumber = 1;
        auto currentIteration = 1;
        auto totalLossForCurrentEpoch = 0.0;
        auto learningRate = baseLearningRate;
        unique_ptr<Matrix> actualMatrix;
        vector<TrainableLayer*> trainableLayers;

        auto expectedMatrix = _context->GetUnique<Matrix>();

        _settings->BatchSize = batchSize;
        _settings->Momentum = momentum;
        _settings->NewBatch = true;

        try {

            for (auto epoch = 0; epoch < epochs; epoch++) {
                for (auto i = 0; i < numberOfExamples; i++) {

                    actualMatrix = _networkRunner->Run(inputLayers[i]);

                    expectedMatrix->Set(actualMatrix->GetRows(), actualMatrix->GetCols(), expectedOutputs[i]);

                    auto loss = _loss->CalculateLoss(*expectedMatrix, *actualMatrix);

                    totalLossForCurrentEpoch += loss;
                    _logger->LogLine("The loss is: " + to_string(loss));

                    auto castOp = [](Layer* layer) {
                        return dynamic_cast<TrainableLayer*>(layer);
                    };

                    vector<Layer*> tempLayers;

                    for (auto& layer : _pyNetwork->Layers) {
                        tempLayers.push_back(layer.get());
                    }

                    auto temp = views::all(tempLayers)
                        | views::filter([castOp](Layer* layer) {return castOp(layer); }) 
                        | views::transform([castOp](Layer* layer) {return castOp(layer); }) 
                        | views::reverse;

                    trainableLayers = vector<TrainableLayer*>(temp.begin(), temp.end());

                    auto lossDerivative = _loss->CalculateDerivative(*expectedMatrix, *actualMatrix);

                    _gradientCalculator->CalculateGradients(trainableLayers, * lossDerivative);

                    if (batchNumber == batchSize) {

                        _logger->LogLine("The learning rate is: " + to_string(learningRate));
                        _trainingAlgorithm->UpdateWeights(trainableLayers, learningRate, false);

                        printf("Iteration %d, Error is %f\n", i, loss);
                        batchNumber = 1;
                        _settings->NewBatch = true;
                    
                    }
                    else {
                        batchNumber++;
                        _settings->NewBatch = false;
                    }

                    currentIteration++;
                }

                if (_vlSettings != nullptr) {

                    auto adjustedTotalLossForCurrentEpoch = 0.0;
                    for (auto j = 0; j < numberOfExamples; j++) {
                        _networkRunner->Run(inputLayers[j]);
                        expectedMatrix->Set(actualMatrix->GetRows(), actualMatrix->GetCols(), expectedOutputs[j]);
                        adjustedTotalLossForCurrentEpoch += _loss->CalculateLoss(*expectedMatrix, *actualMatrix);
                    }

                    if (adjustedTotalLossForCurrentEpoch > (1 + _vlSettings->ErrorThreshold) * totalLossForCurrentEpoch) {
                        learningRate = learningRate * _vlSettings->LRDecrease;
                        _trainingAlgorithm->UpdateWeights(trainableLayers, learningRate, true);
                        _settings->Momentum = 0;
                    }
                    else if (adjustedTotalLossForCurrentEpoch > totalLossForCurrentEpoch) {
                        _settings->Momentum = momentum;
                    }
                    else {
                        learningRate = learningRate * _vlSettings->LRIncrease;
                        _settings->Momentum = momentum;
                    }
                }

                totalLossForCurrentEpoch = 0.0;
            }
        }
        catch (const char* message) {
            _logger->LogLine(message);
            printf(message);
        }
	}

    void NetworkTrainer::SetVLSettings(double errorThreshold, double lrDecrease, double lrIncrease) {
        _vlSettings = make_unique<VariableLearningSettings>();
        _vlSettings->ErrorThreshold = errorThreshold;
        _vlSettings->LRDecrease = lrDecrease;
        _vlSettings->LRIncrease = lrIncrease;
	}
}

