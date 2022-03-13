module;
#include <memory>
#include <string>
export module PyNet.Infrastructure:NetworkTrainer;

import :AdjustmentCalculator;
import :NetworkRunner;
import :GradientCalculator;
import :TrainingAlgorithm;
import :PyNetwork;
import :VariableLearningSettings;
import PyNet.DI;
import PyNet.Models;

using namespace PyNet::DI;
using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	class NetworkTrainer {

	private:

		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
		shared_ptr<Context> _context;
		shared_ptr<NetworkRunner> _networkRunner;
		shared_ptr<GradientCalculator> _gradientCalculator;
		shared_ptr<TrainingAlgorithm> _trainingAlgorithm;
		shared_ptr<ILogger> _logger;
		shared_ptr<PyNetwork> _pyNetwork;
		shared_ptr<Loss> _loss;
		unique_ptr<VariableLearningSettings> _vlSettings = nullptr;

		NetworkTrainer(shared_ptr<AdjustmentCalculator> adjustmentCalculator, shared_ptr<Context> context,
			shared_ptr<NetworkRunner> networkRunner, shared_ptr<GradientCalculator> gradientCalculator,
			shared_ptr<TrainingAlgorithm> trainingAlgirithm, shared_ptr<ILogger> logger, shared_ptr<PyNetwork> pyNetwork,
			shared_ptr<Loss> loss) : _adjustmentCalculator(adjustmentCalculator),
		_context(context), _networkRunner(networkRunner), _gradientCalculator(gradientCalculator), _trainingAlgorithm(trainingAlgirithm),
			_logger(logger), _pyNetwork(pyNetwork), _loss(loss) {}

	public:

		static auto factory(shared_ptr<AdjustmentCalculator> adjustmentCalculator, shared_ptr<Context> context,
			shared_ptr<NetworkRunner> networkRunner, shared_ptr<GradientCalculator> gradientCalculator, shared_ptr<TrainingAlgorithm> trainingAlgorithm,
			shared_ptr<ILogger> logger, shared_ptr<PyNetwork> pyNetwork, shared_ptr<Loss> loss) {
			return new NetworkTrainer(adjustmentCalculator, context, networkRunner, gradientCalculator, trainingAlgorithm, logger, pyNetwork, loss);
		}

		double* TrainNetwork(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate,
			double momentum, int epochs) {
			auto batchNumber = 1;
			auto currentIteration = 1;
			auto totalLossForCurrentEpoch = 0.0;
			auto learningRate = baseLearningRate;

			auto expectedVector = _context->GetUnique<Vector>();

			_adjustmentCalculator->SetBatchSize(batchSize);
			_adjustmentCalculator->SetMomentum(momentum);

			try {

				for (auto epoch = 0; epoch < epochs; epoch++) {
					for (auto i = 0; i < numberOfExamples; i++) {

						_networkRunner->Run(inputLayers[i]);

						expectedVector->Set(_pyNetwork->GetLastLayer().GetRows(), expectedOutputs[i]);

						auto loss = _loss->CalculateLoss(*expectedVector, _pyNetwork->GetLastLayer());

						totalLossForCurrentEpoch += loss;
						_logger->LogLine("The loss is: " + to_string(loss));
						_pyNetwork->Losses.push_back(loss);

						auto lossDerivative = _loss->CalculateDerivative(*expectedVector, _pyNetwork->GetLastLayer());

						_gradientCalculator->CalculateGradients(_pyNetwork->Weights, _pyNetwork->Layers, *expectedVector, *lossDerivative);

						if (batchNumber == batchSize) {

							_logger->LogLine("The learning rate is: " + to_string(learningRate));
							_trainingAlgorithm->UpdateWeights(_pyNetwork->Weights, _pyNetwork->Biases, learningRate, false);

							printf("Iteration %d, Error is %f\n", i, loss);
							batchNumber = 1;
						}
						else {
							batchNumber++;
						}

						currentIteration++;
					}

					if (_vlSettings != nullptr) {

						auto adjustedTotalLossForCurrentEpoch = 0.0;
						for (auto j = 0; j < numberOfExamples; j++) {
							_networkRunner->Run(inputLayers[j]);
							expectedVector->Set(_pyNetwork->GetLastLayer().GetRows(), expectedOutputs[j]);
							adjustedTotalLossForCurrentEpoch += _loss->CalculateLoss(*expectedVector, _pyNetwork->GetLastLayer());
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
				}

			}
			catch (const char* message) {
				_logger->LogLine(message);
			}

			return _pyNetwork->Losses.data();
		}

		__declspec(dllexport) void SetVLSettings(VariableLearningSettings* vlSettings) {
			_vlSettings = unique_ptr<VariableLearningSettings>(vlSettings);
		}
	};
}