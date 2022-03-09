module;
#include <memory>
#include <vector>
export module PyNet.Infrastructure:AdjustmentCalculator;

import :Settings;
import PyNet.Models;
import PyNet.DI;

using namespace std;
using namespace PyNet::DI;
using namespace PyNet::Models;

class AdjustmentCalculator
{
private:
	vector<unique_ptr<Matrix>> _weightAdjustments = vector<unique_ptr<Matrix>>();
	vector<unique_ptr<Vector>> _biasAdjustments = vector<unique_ptr<Vector>>();
	bool _newBatch = true;
	int _batchSize = 0;
	double _momentum = 0;
	shared_ptr<Settings> _settings;
	shared_ptr<Context> _context;
	AdjustmentCalculator(shared_ptr<Settings> settings, shared_ptr<Context> context) : _settings(settings), _context(context) {}

public:

	static auto factory(shared_ptr<Settings> settings, shared_ptr<Context> context) {
		return new AdjustmentCalculator(settings, context);
	}

	void AddMatrix(int rows, int cols) {
		auto weightMatrix = _context->GetUnique<Matrix>();
		weightMatrix->Initialise(rows, cols, false);
		_weightAdjustments.push_back(move(weightMatrix));

		auto biasVector = _context->GetUnique<Vector>();
		biasVector->Initialise(rows, false);
		_biasAdjustments.push_back(move(biasVector));
	}

	void SetBatchSize(int batchSize) { _batchSize = batchSize; }

	void SetMomentum(int momentum) { _momentum = momentum; }

	void AddWeightAdjustment(int matrixIndex, unique_ptr<Matrix> adjustments) {

		auto adjustmentWithMomentum = *adjustments * (1 - _momentum);

		if (_newBatch) {

			if (_momentum > 0) {
				adjustmentWithMomentum = *adjustmentWithMomentum - *(*_weightAdjustments[matrixIndex] * _momentum);
			}

			_weightAdjustments[matrixIndex] = move(adjustmentWithMomentum);
		}
		else {
			*_weightAdjustments[matrixIndex] += *adjustmentWithMomentum;
		}
	}

	void AddBiasAdjustment(int matrixIndex, double adjustment) {

		auto adjustmentWithMomentum = adjustment * (1 - _momentum);

		if (_newBatch) {
			auto totalAdjustment = adjustmentWithMomentum - ((*_biasAdjustments[matrixIndex])[0] * _momentum);
			_biasAdjustments[matrixIndex]->SetValue(totalAdjustment);
		}
		else {
			_biasAdjustments[matrixIndex]->AddValue(adjustmentWithMomentum);
		}
	}

	unique_ptr<Matrix> GetWeightAdjustment(int matrixIndex) const {
		return move(*_weightAdjustments[matrixIndex] / _batchSize);
	}

	unique_ptr<Vector> GetBiasAdjustment(int matrixIndex) const {
		return move(*_biasAdjustments[matrixIndex] / _batchSize);
	}

	void SetNewBatch(bool newBatch) { _newBatch = newBatch; }
};