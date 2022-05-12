#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"
#include "Settings.h"

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class AdjustmentCalculator
	{
	private:
		shared_ptr<Settings> _settings;
		shared_ptr<PyNet::DI::Context> _context;
		AdjustmentCalculator(shared_ptr<Settings> settings, shared_ptr<PyNet::DI::Context> context) : _settings(settings), _context(context) {}

	public:

		static auto factory(shared_ptr<Settings> settings, shared_ptr<PyNet::DI::Context> context) {
			return new AdjustmentCalculator(settings, context);
		}

		void CalculateWeightAdjustment(Matrix& newAdjustment, Matrix& total);
		double CalculateBiasAdjustment(double newAdjustment, double total);
	};
}



