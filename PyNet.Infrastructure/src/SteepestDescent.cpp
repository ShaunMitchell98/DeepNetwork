#include "SteepestDescent.h"
#include <algorithm>

namespace PyNet::Infrastructure {

    void SteepestDescent::UpdateWeights(std::vector<std::unique_ptr<Matrix>>& weightMatrices, vector<unique_ptr<Vector>>& biases, double learningRate, bool reverse) {

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
}

