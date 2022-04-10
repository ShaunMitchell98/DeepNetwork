#pragma once
#include <memory>
#include <string>
#include "AdjustmentCalculator.h"
#include "PyNet.DI/Context.h"
#include "NetworkRunner.h"
#include "GradientCalculator.h"
#include "TrainingAlgorithm.h"
#include "PyNet.Models/ILogger.h"
#include "PyNetwork.h"
#include "PyNet.Models/Loss.h"
#include "VariableLearningSettings.h"

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
			double momentum, int epochs);

		void SetVLSettings(VariableLearningSettings* vlSettings);
	};
}