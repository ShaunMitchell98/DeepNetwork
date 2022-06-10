#pragma once

#include <memory>
#include <vector>
#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"
#include "AdjustmentCalculator.h"
#include "TrainingAlgorithm.h"
#include "PyNet.Models/ILogger.h"
#include "Headers.h"

using namespace PyNet::DI;

namespace PyNet::Infrastructure {

	class EXPORT SteepestDescent : public TrainingAlgorithm
	{
	private:
		shared_ptr<Context> _context;
		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
		shared_ptr<ILogger> _logger;
		SteepestDescent(shared_ptr<Context> context, shared_ptr<ILogger> logger, shared_ptr<AdjustmentCalculator> adjustmentCalculator) :_context(context), _logger(logger), _adjustmentCalculator(adjustmentCalculator) {}
	public:

		static auto factory(shared_ptr<Context> context, shared_ptr<ILogger> logger, shared_ptr<AdjustmentCalculator> adjustmentCalculator) {
			return new SteepestDescent{ context, logger, adjustmentCalculator };
		}

		typedef TrainingAlgorithm base;

		void UpdateWeights(vector<unique_ptr<Matrix>>& weightMatrices, vector<unique_ptr<Vector>>& biases, double learningRate, bool reverse = false) override;
		~SteepestDescent() override = default;
	};

}

