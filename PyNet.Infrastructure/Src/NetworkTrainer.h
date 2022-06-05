#pragma once
#include <memory>
#include <string>
#include <map>
#include "PyNet.DI/Context.h"
#include "NetworkRunner.h"
#include "BackPropagator.h"
#include "TrainingAlgorithm.h"
#include "PyNetwork.h"
#include "PyNet.Models/Loss.h"
#include "VLService.h"
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
		unique_ptr<BackPropagator> _backPropagator;
		shared_ptr<TrainingAlgorithm> _trainingAlgorithm;
		shared_ptr<PyNetwork> _pyNetwork;
		shared_ptr<Loss> _loss;
		unique_ptr<VLService> _vlService;
		shared_ptr<Settings> _settings;

		NetworkTrainer(shared_ptr<Context> context,
			unique_ptr<NetworkRunner> networkRunner, unique_ptr<BackPropagator> backPropagator,
			shared_ptr<TrainingAlgorithm> trainingAlgirithm, shared_ptr<PyNetwork> pyNetwork,
			shared_ptr<Loss> loss, shared_ptr<Settings> settings, unique_ptr<VLService> vlService) :
		_context(context), _networkRunner(move(networkRunner)), _backPropagator(move(backPropagator)), _trainingAlgorithm(trainingAlgirithm),
			_pyNetwork(pyNetwork), _loss(loss), _settings(settings), _vlService(move(vlService)) {}

		void TrainOnExample(shared_ptr<Matrix> input, const Matrix& expectedOutput, int& batchNumber, double& learningRate, double& totalLossForCurrentEpoch, vector<TrainableLayer*> trainableLayers);

	public:

		static auto factory(shared_ptr<Context> context,
			unique_ptr<NetworkRunner> networkRunner, unique_ptr<BackPropagator> backPropagator, shared_ptr<TrainingAlgorithm> trainingAlgorithm,
			shared_ptr<PyNetwork> pyNetwork, shared_ptr<Loss> loss, shared_ptr<Settings> settings, unique_ptr<VLService> vlService) {
			return new NetworkTrainer(context, move(networkRunner), move(backPropagator), trainingAlgorithm, pyNetwork, loss, settings, move(vlService));
		}

		void TrainNetwork(vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>> trainingPairs);
	};
}