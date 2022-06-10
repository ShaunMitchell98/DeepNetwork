#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"
#include "Settings.h"
#include "TrainingState.h"
#include "Headers.h"

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class EXPORT AdjustmentCalculator
	{
	private:
		shared_ptr<Settings> _settings;
		shared_ptr<TrainingState> _trainingState;
		shared_ptr<PyNet::DI::Context> _context;
		AdjustmentCalculator(shared_ptr<Settings> settings, shared_ptr<TrainingState> trainingState, 
			shared_ptr<PyNet::DI::Context> context) : _settings(settings), _trainingState(trainingState), _context(context) {}

	public:

		static auto factory(shared_ptr<Settings> settings, shared_ptr<TrainingState> trainingState, shared_ptr<PyNet::DI::Context> context) {
			return new AdjustmentCalculator(settings, trainingState, context);
		}

		void CalculateWeightAdjustment(Matrix& newAdjustment, Matrix& total);
		double CalculateBiasAdjustment(double newAdjustment, double total);
	};
}



