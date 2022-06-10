#include "GradientCalculator.h"

namespace PyNet::Infrastructure {

	unique_ptr<Matrix> GradientCalculator::CalculateWeightMatrixGradient(const Matrix& layerAboveMatrix, const Vector& inputLayer, Vector& outputLayer,
		Vector& dLoss_dLayerAbove) {
		auto dLoss_dActivatedLayerMatrix = *~layerAboveMatrix * dLoss_dLayerAbove;
		auto dLoss_dActivatedLayer = _context->GetUnique<Vector>();
		*dLoss_dActivatedLayer = move(*dLoss_dActivatedLayerMatrix);

		auto dLoss_dLayer = *dLoss_dActivatedLayer ^ *outputLayer.CalculateActivationDerivative();

		auto dLoss_dWeight = *dLoss_dLayer * *~inputLayer;

		dLoss_dLayerAbove = move(*dLoss_dLayer);

		return move(dLoss_dWeight);
	}

	void GradientCalculator::CalculateGradients(const vector<unique_ptr<Matrix>>& weightMatrices,
		const vector<unique_ptr<Vector>>& layers, const Vector& expectedLayer, const Vector& lossDerivative) {

		auto dActivatedLayerAbove_dLayerAbove = layers[layers.size() - 1]->CalculateActivationDerivative();
		auto dLoss_dLayerAbove = lossDerivative ^ *dActivatedLayerAbove_dLayerAbove;
		_adjustmentCalculator->AddWeightAdjustment(weightMatrices.size() - 1, move(*dLoss_dLayerAbove * *~*layers[layers.size() - 2]));
		_adjustmentCalculator->AddBiasAdjustment(weightMatrices.size() - 1, *dLoss_dLayerAbove | *dActivatedLayerAbove_dLayerAbove);

		for (int i = weightMatrices.size() - 2; i >= 0; i--) {

			_adjustmentCalculator->AddWeightAdjustment(i, std::move(CalculateWeightMatrixGradient(*weightMatrices[i + 1.0], *layers[i], *layers[i + 1.0], *dLoss_dLayerAbove)));
			_adjustmentCalculator->AddBiasAdjustment(i, CalculateBiasGradient(*weightMatrices[i + 1.0], *layers[i], *layers[i + 1.0], *dLoss_dLayerAbove));
		}

		//_logger->LogLine("Calculated gradient.");

		_adjustmentCalculator->SetNewBatch(false);
	}

    double GradientCalculator::CalculateBiasGradient(const Matrix& layerAboveMatrix, const Vector& inputLayer, Vector& outputLayer, Vector& dLoss_dLayerAbove) {

        return dLoss_dLayerAbove | *outputLayer.CalculateActivationDerivative();
    }
}
