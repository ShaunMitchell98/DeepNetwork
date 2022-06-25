#pragma once

#include <memory>
#include <vector>
#include "TrainingAlgorithm.h"
#include "Headers.h"
#include "Settings.h"
#include "PyNet.Models/ILogger.h"

using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure {

	class EXPORT SteepestDescent : public TrainingAlgorithm
	{
	private:
		shared_ptr<Settings> _settings;
		shared_ptr<ILogger> _logger;

		SteepestDescent(shared_ptr<Settings> settings, shared_ptr<ILogger> logger) :
			_settings(settings), _logger(logger) {}
	public:

		static auto factory(shared_ptr<Settings> settings, shared_ptr<ILogger> logger) {
			return new SteepestDescent{ settings, logger };
		}

		typedef TrainingAlgorithm base;

		void UpdateWeights(vector<TrainableLayer*> layers, double learningRate, bool reverse = false) const override;
		~SteepestDescent() override = default;
	};

}

