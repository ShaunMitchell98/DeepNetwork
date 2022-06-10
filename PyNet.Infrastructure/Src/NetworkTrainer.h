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
#include "TrainingState.h"
#include "Settings.h"
#include "Headers.h"

using namespace PyNet::DI;
using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkTrainer {

	private:

		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
		shared_ptr<Context> _context;
		shared_ptr<NetworkRunner> _networkRunner;
		shared_ptr<GradientCalculator> _gradientCalculator;
		shared_ptr<TrainingAlgorithm> _trainingAlgorithm;
		shared_ptr<ILogger> _logger;
		shared_ptr<PyNetwork> _pyNetwork;
		shared_ptr<Loss> _loss;
		shared_ptr<TrainingState> _trainingState;
		shared_ptr<Settings> _settings;
		unique_ptr<VariableLearningSettings> _vlSettings;

		NetworkTrainer(shared_ptr<AdjustmentCalculator> adjustmentCalculator, shared_ptr<Context> context,
			shared_ptr<NetworkRunner> networkRunner, shared_ptr<GradientCalculator> gradientCalculator,
			shared_ptr<TrainingAlgorithm> trainingAlgirithm, shared_ptr<ILogger> logger, shared_ptr<PyNetwork> pyNetwork,
			shared_ptr<Loss> loss, shared_ptr<TrainingState> trainingState, shared_ptr<Settings> settings) : _adjustmentCalculator(adjustmentCalculator),
			_context(context), _networkRunner(networkRunner), _gradientCalculator(gradientCalculator), _trainingAlgorithm(trainingAlgirithm),
			_logger(logger), _pyNetwork(pyNetwork), _loss(loss), _trainingState(trainingState), _settings(settings)
		{}

	public:

		static auto factory(shared_ptr<AdjustmentCalculator> adjustmentCalculator, shared_ptr<Context> context,
			shared_ptr<NetworkRunner> networkRunner, shared_ptr<GradientCalculator> gradientCalculator, shared_ptr<TrainingAlgorithm> trainingAlgorithm,
			shared_ptr<ILogger> logger, shared_ptr<PyNetwork> pyNetwork, shared_ptr<Loss> loss, shared_ptr<TrainingState> trainingState, shared_ptr<Settings> settings) {
			return new NetworkTrainer(adjustmentCalculator, context, networkRunner, gradientCalculator, trainingAlgorithm, logger, pyNetwork, loss, trainingState, settings);
		}

		double* TrainNetwork(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate,
			double momentum, int epochs);

		void SetVLSettings(double errorThreshold, double lrDecrease, double lrIncrease);
	};
}