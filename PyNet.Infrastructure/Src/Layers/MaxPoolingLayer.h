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

		MaxPoolingLayer(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder) : _receptiveFieldProvider{ receptiveFieldProvider },
			_matrixPadder{ matrixPadder } {}

		double Maximum(const Matrix& input) const {

			
		}
	public:

		static auto factory(shared_ptr<ReceptiveFieldProvider> receptiveFieldProvider, shared_ptr<MatrixPadder> matrixPadder) {
			return new MaxPoolingLayer(receptiveFieldProvider, matrixPadder);
		}

		void Initialise(int filterSize) {
			_filterSize = filterSize;
		}

		unique_ptr<Matrix> Apply(unique_ptr<Matrix> input) override {

			auto paddedMatrix = _matrixPadder->PadMatrix(*input, _filterSize);

			auto featureMap = input->Copy();

			for (auto row = 0; row < featureMap->GetRows(); row++) {
				for (auto col = 0; col < featureMap->GetCols(); col++) {

					auto receptiveField = _receptiveFieldProvider->GetReceptiveField(*paddedMatrix, _filterSize);

					auto values = receptiveField->GetCValues();

					(*featureMap)(row, col) = *ranges::max_element(values);
				}
			}

			return featureMap;
		}

		unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const {
			throw "To do";
		}

		size_t GetRows() const override {
			return 0;
		}

		size_t GetCols() const override {
			return 0;
		}
	};
}