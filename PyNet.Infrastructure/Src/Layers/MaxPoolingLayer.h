#pragma once

#include "Layer.h"
#include "ReceptiveFieldProvider.h"
#include "MatrixPadder.h"
#include <algorithm>
#include <ranges>

using namespace std;

namespace PyNet::Infrastructure::Layers {
	class MaxPoolingLayer : public Layer {
	private:

		int _filterSize = 0;
		shared_ptr<ReceptiveFieldProvider> _receptiveFieldProvider;
		shared_ptr<MatrixPadder> _matrixPadder;

		MaxPoolingLayer(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder,
			unique_ptr<Matrix> input, unique_ptr<Matrix> output) : _receptiveFieldProvider{ receptiveFieldProvider },
			_matrixPadder{ matrixPadder }, Layer(move(input), move(output)) {}

	public:

		static auto factory(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder, unique_ptr<Matrix> input, 
			unique_ptr<Matrix> output) {
			return new MaxPoolingLayer(receptiveFieldProvider, matrixPadder, move(input), move(output));
		}

		void Initialise(int filterSize, int rows, int cols) {
			_filterSize = filterSize;
			Input->Initialise(rows, cols, false);
			Output->Initialise(rows, cols, false);
		}

		shared_ptr<Matrix> ApplyInternal(shared_ptr<Matrix> input) override {

			auto paddedMatrix = _matrixPadder->PadMatrix(*input, _filterSize);
			Input = paddedMatrix;

			Output = input->Copy();
			
			for (size_t row = 1; row <= input->GetRows(); row++) {
				for (size_t col = 1; col <= input->GetCols(); col++) {

					auto receptiveField = _receptiveFieldProvider->GetReceptiveField(*paddedMatrix, _filterSize, row, col);

					auto& values = receptiveField->GetCValues();

					#ifdef _WIN32
					(*Output)(row, col) = *ranges::max_element(values);
					#else

					auto largest = values.front();

					for (auto& element : values) 
					{
						if (element > largest) 
						{
							largest = element;
						}
					}

					(*Output)(row, col) = largest;
					#endif
				}
			}

			return Output;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const override {

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

							if ((*Output)(outputRow, outputCol) == (*Input)(inputRow, inputCol)) {
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