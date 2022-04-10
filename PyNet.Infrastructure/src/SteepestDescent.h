#pragma once

#include <memory>
#include <vector>
#include "PyNet.Models/Matrix.h"
#include "PyNet.DI/Context.h"
#include "AdjustmentCalculator.h"
#include "TrainingAlgorithm.h"

using namespace PyNet::DI;

namespace PyNet::Infrastructure {

	class SteepestDescent : public TrainingAlgorithm
	{
	private:
		shared_ptr<Context> _context;
		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
		SteepestDescent(shared_ptr<Context> context, shared_ptr<AdjustmentCalculator> adjustmentCalculator) :_context(context), _adjustmentCalculator(adjustmentCalculator) {}
	public:

		static auto factory(shared_ptr<Context> context, shared_ptr<AdjustmentCalculator> adjustmentCalculator) {
			return new SteepestDescent{ context, adjustmentCalculator };
		}

		typedef TrainingAlgorithm base;

		void UpdateWeights(vector<unique_ptr<Matrix>>& weightMatrices, vector<unique_ptr<Vector>>& biases, double learningRate, bool reverse = false) override;
	};

}

