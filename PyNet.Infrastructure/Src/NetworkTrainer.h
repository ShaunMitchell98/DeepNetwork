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
#include "TrainingState.h"
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
		shared_ptr<TrainingState> _trainingState;
		int _batchNumber;
		double _learningRate;
		double _totalLossForCurrentEpoch;

		NetworkTrainer(shared_ptr<Context> context,
			unique_ptr<NetworkRunner> networkRunner, unique_ptr<BackPropagator> backPropagator,
			shared_ptr<TrainingAlgorithm> trainingAlgirithm, shared_ptr<PyNetwork> pyNetwork,
			shared_ptr<Loss> loss, shared_ptr<Settings> settings, unique_ptr<VLService> vlService, shared_ptr<TrainingState> trainingState) :
		_context(context), _networkRunner(move(networkRunner)), _backPropagator(move(backPropagator)), _trainingAlgorithm(trainingAlgirithm),
			_pyNetwork(pyNetwork), _loss(loss), _settings(settings), _vlService(move(vlService)), _trainingState(trainingState), _batchNumber{ 1 }, _learningRate{ 0 }, _totalLossForCurrentEpoch{ 0.0 } {}

		void TrainOnExample(shared_ptr<Matrix> input, const Matrix& expectedOutput, vector<TrainableLayer*> trainableLayers);

	public:

		static auto factory(shared_ptr<Context> context,
			unique_ptr<NetworkRunner> networkRunner, unique_ptr<BackPropagator> backPropagator, shared_ptr<TrainingAlgorithm> trainingAlgorithm,
			shared_ptr<PyNetwork> pyNetwork, shared_ptr<Loss> loss, shared_ptr<Settings> settings, unique_ptr<VLService> vlService, shared_ptr<TrainingState> trainingState) {
			return new NetworkTrainer(context, move(networkRunner), move(backPropagator), trainingAlgorithm, pyNetwork, loss, settings, move(vlService),
				trainingState);
		}

		void TrainNetwork(vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>> trainingPairs);
	};
}