#pragma once

#include <memory>
#include <vector>
#include "TrainingAlgorithm.h"
#include "Headers.h"
#include "Settings.h"

using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure {

	class EXPORT SteepestDescent : public TrainingAlgorithm
	{
	private:
		shared_ptr<Settings> _settings;

		SteepestDescent(shared_ptr<Settings> settings) :
			_settings(settings) {}
	public:

		static auto factory(shared_ptr<Settings> settings) {
			return new SteepestDescent{ settings };
		}

		typedef TrainingAlgorithm base;

		void UpdateWeights(vector<TrainableLayer*> layers, double learningRate, bool reverse = false) const override;
		~SteepestDescent() override = default;
	};

}

