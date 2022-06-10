#include "NetworkTrainer.h"
#include <format>

namespace PyNet::Infrastructure {
    
    double* NetworkTrainer::TrainNetwork(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate, double momentum, int epochs) {
        auto batchNumber = 1;
        auto currentIteration = 1;
        auto totalLossForCurrentEpoch = 0.0;
        auto learningRate = baseLearningRate;

        auto expectedVector = _context->GetUnique<Vector>();

        _adjustmentCalculator->SetBatchSize(batchSize);
        _adjustmentCalculator->SetMomentum(momentum);

        _trainingState->ExampleNumber = 1;

        try {

            for (_trainingState->Epoch = 1; _trainingState->Epoch <= epochs; _trainingState->Epoch++) {
                for (auto i = 0; i < numberOfExamples; i++) {

                    _networkRunner->Run(inputLayers[i]);

                    expectedVector->Set(_pyNetwork->GetOutputLayer().GetRows(), expectedOutputs[i]);

                    auto loss = _loss->CalculateLoss(*expectedVector, _pyNetwork->GetOutputLayer());

                    totalLossForCurrentEpoch += loss;
                    _logger->LogLine("Loss is {}", make_format_args(loss));
                    _pyNetwork->Losses.push_back(loss);

                    auto lossDerivative = _loss->CalculateDerivative(*expectedVector, _pyNetwork->GetOutputLayer());

                    _gradientCalculator->CalculateGradients(_pyNetwork->Weights, _pyNetwork->Layers, *expectedVector, *lossDerivative);

                    if (batchNumber == batchSize) {

                        //_logger->LogLine("The learning rate is: " + to_string(learningRate));
                        _trainingAlgorithm->UpdateWeights(_pyNetwork->Weights, _pyNetwork->Biases, learningRate, false);

                        printf("Epoch %d, Iteration %d, Error is %f\n", _trainingState->Epoch, i + _settings->StartExampleNumber, loss);
                        batchNumber = 1;
                    }
                    else {
                        batchNumber++;
                        }

                    currentIteration++;
                    _trainingState->ExampleNumber++;
                }

                if (_vlSettings != nullptr) {

                    auto adjustedTotalLossForCurrentEpoch = 0.0;
                    for (auto j = 0; j < numberOfExamples; j++) {
                        _networkRunner->Run(inputLayers[j]);
                        expectedVector->Set(_pyNetwork->GetOutputLayer().GetRows(), expectedOutputs[j]);
                        adjustedTotalLossForCurrentEpoch += _loss->CalculateLoss(*expectedVector, _pyNetwork->GetOutputLayer());
                    }

                    if (adjustedTotalLossForCurrentEpoch > (1 + _vlSettings->ErrorThreshold) * totalLossForCurrentEpoch) {
                        learningRate = learningRate * _vlSettings->LRDecrease;
                        _trainingAlgorithm->UpdateWeights(_pyNetwork->Weights, _pyNetwork->Layers, learningRate, true);
                        _adjustmentCalculator->SetMomentum(0);
                    }
                    else if (adjustedTotalLossForCurrentEpoch > totalLossForCurrentEpoch) {
                        _adjustmentCalculator->SetMomentum(momentum);
                    }
                    else {
                        learningRate = learningRate * _vlSettings->LRIncrease;
                        _adjustmentCalculator->SetMomentum(momentum);
                    }
                }

                totalLossForCurrentEpoch = 0.0;
                _trainingState->ExampleNumber = 0;
            }
        }
        catch (const char* message) {
            _logger->LogLine(message);
        }

        return _pyNetwork->Losses.data();
	}

    void NetworkTrainer::SetVLSettings(double errorThreshold, double lrDecrease, double lrIncrease) {
        _vlSettings = make_unique<VariableLearningSettings>();
        _vlSettings->ErrorThreshold = errorThreshold;
        _vlSettings->LRDecrease = lrDecrease;
        _vlSettings->LRIncrease = lrIncrease;
	}
}

