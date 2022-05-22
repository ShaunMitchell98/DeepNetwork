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
			_dLoss_dWeightSum = context->GetUnique<Matrix>();
			_dLoss_dBiasSum = 0;
		}

		unique_ptr<Matrix> dLoss_dWeight(const Matrix& dLoss_dOutput) const 
		{
			auto dLoss_dWeight = _weights->Copy();
			double sum;

			for (size_t weightRow = 1; weightRow <= _weights->GetRows(); weightRow++) {
				for (size_t weightCol = 1; weightCol <= _weights->GetCols(); weightCol++) {

					sum = 0.0;
					auto maxRow = Input->GetRows() - _filterSize + weightRow;
					auto maxCol = Input->GetCols() - _filterSize + weightCol;

					for (size_t inputRow = 1; inputRow <= Output->GetRows(); inputRow++) {
						for (size_t inputCol = 1; inputCol <= Output->GetRows(); inputCol++) {
							size_t inputIndexRow = inputRow + weightRow - 1;
							size_t inputIndexCol = inputCol + weightCol - 1;
							sum += (*Input)(inputIndexRow, inputIndexCol) * dLoss_dOutput(inputRow, inputCol);
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

		void Initialise(int filterSize) {
			_weights->Initialise(filterSize, filterSize, true);
			_filterSize = filterSize;
			_dLoss_dWeightSum->Initialise(filterSize, filterSize, false);
		}

		static auto factory(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder, shared_ptr<Context> context,
			shared_ptr<AdjustmentCalculator> adjustmentCalculator) {
			return new ConvolutionalLayer(receptiveFieldProvider, matrixPadder, context, adjustmentCalculator);
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override
		{
			auto paddedMatrix = _matrixPadder->PadMatrix(*input, _filterSize);
			Input = paddedMatrix;

			Output = input->Copy();

			auto maxRows = Input->GetRows() - _filterSize;
			auto maxCols = Input->GetCols() - _filterSize;

			for (size_t row = 1; row <= maxRows; row++) {
				for (size_t col = 1; col <= maxCols; col++) {

					auto receptiveField = _receptiveFieldProvider->GetReceptiveField(*Input, _filterSize, row, col);
					(*Output)(row, col) = (*receptiveField | *_weights) + _bias;
				}
			}

			return Output;
		}	

		void UpdateAdjustments(const Matrix& dLoss_dOutput) override 
		{
			_dLoss_dBiasSum = _adjustmentCalculator->CalculateBiasAdjustment(dLoss_dBias(dLoss_dOutput), _dLoss_dBiasSum);
			_adjustmentCalculator->CalculateWeightAdjustment(*dLoss_dWeight(dLoss_dOutput), *_dLoss_dWeightSum);
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override 
		{
			auto dLoss_dInput = dLoss_dOutput.Copy();
			dLoss_dInput->Initialise(Input->GetRows(), Input->GetCols(), false);
			double sum;
			auto padding = static_cast<double>((_filterSize + 1) / 2);

			for (size_t inputRow = 1; inputRow <= dLoss_dInput->GetRows(); inputRow++) 
			{
				for (size_t inputCol = 1; inputCol <= dLoss_dInput->GetCols(); inputCol++) 
				{
					sum = 0.0;

					auto outputStartRow = max<size_t>(1, inputRow - padding);
					auto outputEndRow = min<size_t>(inputRow, Output->GetRows());

					auto outputStartCol = max<size_t>(1, inputCol - padding);
					auto outputEndCol = min<size_t>(inputCol, Output->GetCols());

					for (size_t outputRow = outputStartRow; outputRow <= outputEndRow; outputRow++) 
					{
						for (size_t outputCol = outputStartCol; outputCol <= outputEndCol; outputCol++)
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