#pragma once

#include <vector>
#include <memory>
#include <map>
#include "PyNet.Models/Matrix.h"
#include "NetworkRunner.h"
#include "PyNet.Models/Loss.h"
#include "VariableLearningSettings.h"
#include "TrainingAlgorithm.h"
#include "Settings.h"

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure
{
	class VLService 
	{
		private:

		unique_ptr<NetworkRunner> _networkRunner;
		unique_ptr<Loss> _loss;
		shared_ptr<VariableLearningSettings> _vlSettings;
		shared_ptr<Settings> _settings;
		unique_ptr<TrainingAlgorithm> _trainingAlgorithm;
		shared_ptr<PyNetwork> _pyNetwork;

		VLService(unique_ptr<NetworkRunner> networkRunner, unique_ptr<Loss> loss, shared_ptr<VariableLearningSettings> vlSettings,
			shared_ptr<Settings> settings, unique_ptr<TrainingAlgorithm> trainingAlgorithm, shared_ptr<PyNetwork> pyNetwork)
			: _networkRunner{ move(networkRunner) }, _loss{ move(loss) }, _vlSettings{ vlSettings }, _settings{ settings }, _trainingAlgorithm(move(trainingAlgorithm)),
		_pyNetwork(pyNetwork) {}

		public:

		static auto factory(unique_ptr<NetworkRunner> networkRunner, unique_ptr<Loss> loss, shared_ptr<VariableLearningSettings> vlSettings,
			shared_ptr<Settings> settings, unique_ptr<TrainingAlgorithm> trainingAlgorithm, shared_ptr<PyNetwork> pyNetwork)
		{
			return new VLService(move(networkRunner), move(loss), vlSettings, settings, move(trainingAlgorithm), pyNetwork);
		}

		void RunVariableLearning(vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>> trainingPairs, double& learningRate, double totalLossForCurrentEpoch);
	};
}