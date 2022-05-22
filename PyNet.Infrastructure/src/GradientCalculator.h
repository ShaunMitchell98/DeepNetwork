#pragma once
#include <memory>
#include <vector>
#include "PyNet.DI/Context.h"
#include "AdjustmentCalculator.h"
#include "Layers/Layer.h"

using namespace std;
using namespace PyNet::Models;
using namespace PyNet::DI;
using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure {

	class GradientCalculator {
	private:
		shared_ptr<Context> _context;
		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;

		GradientCalculator(shared_ptr<Context> context, shared_ptr<AdjustmentCalculator> adjustmentCalculator) :
			_context{ context }, _adjustmentCalculator{ adjustmentCalculator } {}

	public:

		static auto factory(shared_ptr<Context> context, shared_ptr<AdjustmentCalculator> adjustmentCalculator) {
			return new GradientCalculator{ context, adjustmentCalculator };
		}

		void CalculateGradients(vector<Layer*> layers, Matrix& lossDerivative);
	};
}
