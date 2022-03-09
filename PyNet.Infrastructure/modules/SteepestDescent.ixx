module;
#include <memory>
#include <vector>
export module PyNet.Infrastructure:SteepestDescent;

import :AdjustmentCalculator;
import :TrainingAlgorithm;
import PyNet.Models;
import PyNet.DI;

using namespace std;
using namespace PyNet::DI;
using namespace PyNet::Models;

export namespace PyNet::Infrastructure {
	class __declspec(dllexport) SteepestDescent : public TrainingAlgorithm
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

		void UpdateWeights(vector<unique_ptr<Matrix>>& weightMatrices, vector<unique_ptr<Vector>>& biases, double learningRate, bool reverse = false) override {

			auto biasAdjustmentVector = _context->GetUnique<Vector>();

			for (int index = weightMatrices.size() - 1; index >= 0; index--) {

				auto biasAdjustmentMatrix = *_adjustmentCalculator->GetBiasAdjustment(index) * learningRate;
				biasAdjustmentVector->Set(biasAdjustmentMatrix->GetRows(), biasAdjustmentMatrix->Values.data());

				if (reverse) {
					weightMatrices[index] = *weightMatrices[index] + *(*_adjustmentCalculator->GetWeightAdjustment(index) * learningRate);
					biases[index] = *biases[index] + *biasAdjustmentVector;
				}
				else {
					weightMatrices[index] = *weightMatrices[index] - *(*_adjustmentCalculator->GetWeightAdjustment(index) * learningRate);
					biases[index] = *biases[index] - *biasAdjustmentVector;
				}
			}

			_adjustmentCalculator->SetNewBatch(true);
		}
	};
}