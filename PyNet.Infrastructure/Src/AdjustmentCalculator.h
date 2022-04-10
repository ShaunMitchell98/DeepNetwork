#pragma once

#include <memory>
#include "PyNet.Models/Vector.h"
#include "PyNet.DI/Context.h"
#include "Settings.h"

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class AdjustmentCalculator
	{
	private:
		vector<unique_ptr<Matrix>> _weightAdjustments = vector<unique_ptr<Matrix>>();
		vector<unique_ptr<Vector>> _biasAdjustments = vector<unique_ptr<Vector>>();
		bool _newBatch = true;
		int _batchSize = 0;
		double _momentum = 0;
		shared_ptr<Settings> _settings;
		shared_ptr<PyNet::DI::Context> _context;
		AdjustmentCalculator(shared_ptr<Settings> settings, shared_ptr<PyNet::DI::Context> context) : _settings(settings), _context(context) {}

	public:

		static auto factory(shared_ptr<Settings> settings, shared_ptr<PyNet::DI::Context> context) {
			return new AdjustmentCalculator(settings, context);
		}

		void AddMatrix(int rows, int cols);
		void SetBatchSize(int batchSize);
		void SetMomentum(int momentum);
		void AddWeightAdjustment(int matrixIndex, unique_ptr<Matrix> adjustments);
		void AddBiasAdjustment(int matrixIndex, double adjustment);
		unique_ptr<Matrix> GetWeightAdjustment(int matrixIndex) const;
		unique_ptr<Vector> GetBiasAdjustment(int matrixIndex) const;
		void SetNewBatch(bool newBatch);
	};
}



