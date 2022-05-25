#pragma once
#include <memory>
#include <string>
#include "PyNet.DI/Context.h"
#include "NetworkRunner.h"
#include "GradientCalculator.h"
#include "TrainingAlgorithm.h"
#include "PyNet.Models/ILogger.h"
#include "PyNetwork.h"
#include "PyNet.Models/Loss.h"
#include "VariableLearningSettings.h"
#include "Settings.h"
#include "Headers.h"

using namespace PyNet::DI;
using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkTrainer {

	private:

		shared_ptr<Context> _context;
		unique_ptr<NetworkRunner> _networkRunner;
		unique_ptr<GradientCalculator> _gradientCalculator;
		shared_ptr<TrainingAlgorithm> _trainingAlgorithm;
		shared_ptr<ILogger> _logger;
		shared_ptr<PyNetwork> _pyNetwork;
		shared_ptr<Loss> _loss;
		unique_ptr<VariableLearningSettings> _vlSettings;
		shared_ptr<Settings> _settings;

		NetworkTrainer(shared_ptr<Context> context,
			unique_ptr<NetworkRunner> networkRunner, unique_ptr<GradientCalculator> gradientCalculator,
			shared_ptr<TrainingAlgorithm> trainingAlgirithm, shared_ptr<ILogger> logger, shared_ptr<PyNetwork> pyNetwork,
			shared_ptr<Loss> loss, shared_ptr<Settings> settings) :
		_context(context), _networkRunner(move(networkRunner)), _gradientCalculator(move(gradientCalculator)), _trainingAlgorithm(trainingAlgirithm),
			_logger(logger), _pyNetwork(pyNetwork), _loss(loss), _settings(settings) {}

	public:

		static auto factory(shared_ptr<Context> context,
			unique_ptr<NetworkRunner> networkRunner, unique_ptr<GradientCalculator> gradientCalculator, shared_ptr<TrainingAlgorithm> trainingAlgorithm,
			shared_ptr<ILogger> logger, shared_ptr<PyNetwork> pyNetwork, shared_ptr<Loss> loss, shared_ptr<Settings> settings) {
			return new NetworkTrainer(context, move(networkRunner), move(gradientCalculator), trainingAlgorithm, logger, pyNetwork, loss, settings);
		}

		void TrainNetwork(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate,
			double momentum, int epochs);

		void SetVLSettings(double errorThreshold, double lrDecrease, double lrIncrease);
	};
}