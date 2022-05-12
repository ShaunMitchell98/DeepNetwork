#pragma once

#include "TrainableLayer.h"
#include "../ReceptiveFieldProvider.h"
#include "../MatrixPadder.h"
#include "PyNet.DI/Context.h"
#include "AdjustmentCalculator.h"

using namespace PyNet::DI;

namespace PyNet::Infrastructure::Layers {

	class ConvolutionalLayer : public TrainableLayer {
	private:

		int _filterSize = 0;
		unique_ptr<Matrix> _weights;
		unique_ptr<Matrix> _dLoss_dWeightSum;
		unique_ptr<Matrix> _input;
		double _bias = 0;
		double _dLoss_dBiasSum;
		shared_ptr<ReceptiveFieldProvider> _receptiveFieldProvider;
		shared_ptr<MatrixPadder> _matrixPadder;
		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;

		ConvolutionalLayer(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder, shared_ptr<Context> context,
			shared_ptr<AdjustmentCalculator> adjustmentCalculator) :
			_receptiveFieldProvider{ receptiveFieldProvider }, _matrixPadder{ matrixPadder }, _adjustmentCalculator{ adjustmentCalculator } 
		{
			_weights = context->GetUnique<Matrix>();
		}

		unique_ptr<Matrix> dLoss_dWeight(const Matrix& dLoss_dOutput) const 
		{
			auto dLoss_dWeight = _weights->Copy();
			double sum;

			for (size_t weightRow = 0; weightRow < _weights->GetRows(); weightRow++) {
				for (size_t weightCol = 0; weightCol < _weights->GetCols(); weightCol++) {

					sum = 0.0;
					for (size_t inputRow = 0; inputRow < dLoss_dOutput.GetRows(); inputRow++) {
						for (size_t inputCol = 0; inputCol < dLoss_dOutput.GetCols(); inputCol++) {
							sum += (*_input)(inputRow + weightRow - 1, inputCol + weightCol - 1) * dLoss_dOutput(inputRow, inputCol);
						}
					}

					(*dLoss_dWeight)(weightRow, weightCol) = sum;
				}
			}

			return dLoss_dWeight;
		}

		double dLoss_dBias(const Matrix& dLoss_dOutput) const
		{
			auto dLoss_dBias = 0.0;

			for (const auto& element : dLoss_dOutput) 
			{
				dLoss_dBias += element;
			}

			return dLoss_dBias;
		}

	public:

		size_t GetRows() const override {
			return _weights->GetRows();
		}

		size_t GetCols() const override {
			return _weights->GetCols();
		}

		void Initialise(int filterSize) {
			_weights->Initialise(filterSize, filterSize, true);
			_filterSize = filterSize;
		}

		static auto factory(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder, shared_ptr<Context> context,
			shared_ptr<AdjustmentCalculator> adjustmentCalculator) {
			return new ConvolutionalLayer(receptiveFieldProvider, matrixPadder, context, adjustmentCalculator);
		}

		unique_ptr<Matrix> Apply(unique_ptr<Matrix> input) override
		{
			_input.swap(input);

			auto paddedMatrix = _matrixPadder->PadMatrix(*_input, _filterSize);

			auto featureMap = _input->Copy();

			for (auto& element : *featureMap) 
			{
				auto receptiveField = _receptiveFieldProvider->GetReceptiveField(*paddedMatrix, _filterSize);

				element = (*receptiveField | *_weights) + _bias;
			}

			return featureMap;
		}	

		void UpdateAdjustments(const Matrix& dLoss_dOutput) override 
		{
			_dLoss_dBiasSum = _adjustmentCalculator->CalculateBiasAdjustment(dLoss_dBias(dLoss_dOutput), _dLoss_dBiasSum);
			_adjustmentCalculator->CalculateWeightAdjustment(*dLoss_dWeight(dLoss_dOutput), *_dLoss_dWeightSum);
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override 
		{
			auto dLoss_dInput = dLoss_dOutput.Copy();
			dLoss_dInput->Initialise(_input->GetRows(), _input->GetCols(), false);
			double sum;

			for (size_t inputRow = 0; inputRow < dLoss_dInput->GetRows(); inputRow++) 
			{
				for (size_t inputCol = 0; inputCol < dLoss_dInput->GetCols(); inputCol++) 
				{
					sum = 0.0;
					for (size_t outputRow = 0; outputRow < dLoss_dOutput.GetRows(); outputRow++) 
					{
						for (size_t outputCol = 0; outputCol < dLoss_dOutput.GetCols(); outputCol++)
						{
							sum += dLoss_dOutput(outputRow, outputCol) * (*_weights)(inputRow + 1 - outputRow, inputCol + 1 - outputCol);
						}
					}

					(*dLoss_dInput)(inputRow, inputCol) = sum;
				}
			}

			return dLoss_dInput;
		}

		Matrix& GetdLoss_dWeightSum() const override 
		{
			return *_dLoss_dWeightSum;
		}

		double GetdLoss_dBiasSum() const override
		{
			return _dLoss_dBiasSum;
		}

		Matrix& GetWeights() override
		{
			return *_weights;
		}

		double& GetBias() override
		{
			return _bias;
		}
	};
}