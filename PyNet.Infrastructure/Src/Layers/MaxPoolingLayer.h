#pragma once

#include "Layer.h"
#include "ReceptiveFieldProvider.h"
#include "MatrixPadder.h"
#include <algorithm>

namespace PyNet::Infrastructure::Layers {
	class MaxPoolingLayer : public Layer {
	private:

		int _filterSize = 0;
		shared_ptr<ReceptiveFieldProvider> _receptiveFieldProvider;
		shared_ptr<MatrixPadder> _matrixPadder;

		MaxPoolingLayer(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder,
			unique_ptr<Matrix> input) : _receptiveFieldProvider{ receptiveFieldProvider },
			_matrixPadder{ matrixPadder }, Layer(move(input)) {}

	public:

		static auto factory(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder, unique_ptr<Matrix> input) {
			return new MaxPoolingLayer(receptiveFieldProvider, matrixPadder, move(input));
		}

		void Initialise(int filterSize) {
			_filterSize = filterSize;
		}

		shared_ptr<Matrix> Apply(shared_ptr<Matrix> input) override {

			auto paddedMatrix = _matrixPadder->PadMatrix(*input, _filterSize);
			Input = paddedMatrix;

			Output = input->Copy();

			auto maxRows = Input->GetRows() - _filterSize;
			auto maxCols = Input->GetCols() - _filterSize;
			
			for (size_t row = 1; row <= maxRows; row++) {
				for (size_t col = 1; col <= maxCols; col++) {

					auto receptiveField = _receptiveFieldProvider->GetReceptiveField(*paddedMatrix, _filterSize, row, col);

					auto& values = receptiveField->GetCValues();

					(*Output)(row, col) = *ranges::max_element(values);
				}
			}

			return Output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const {

			auto dLoss_dInput = dLoss_dOutput.Copy();

			auto padding = static_cast<double>((_filterSize - 1) / 2);

			for (size_t inputRow = 1; inputRow <= dLoss_dInput->GetRows(); inputRow++) {
				for (size_t inputCol = 1; inputCol <= dLoss_dInput->GetCols(); inputCol++) {
				
					auto sum = 0.0;

					auto outputStartRow = max<size_t>(1, inputRow - padding);
					auto outputEndRow = min<size_t>(inputRow, Output->GetRows());

					auto outputStartCol = max<size_t>(1, inputCol - padding);
					auto outputEndCol = min<size_t>(inputCol, Output->GetCols());

					for (size_t outputRow = outputStartRow; outputRow <= outputEndRow; outputRow++) {
						for (size_t outputCol = outputStartCol; outputCol <= outputEndCol; outputCol++) {

							if ((*Output)(outputRow, outputCol) = (*Input)(inputRow, inputCol)) {
								sum += dLoss_dOutput(outputRow, outputCol);
							}
						}
					}

					(*dLoss_dInput)(inputRow, inputCol) = sum;
				}
			}

			return dLoss_dInput;
		}
	};
}