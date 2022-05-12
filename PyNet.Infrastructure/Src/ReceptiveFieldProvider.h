#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	class ReceptiveFieldProvider {
	public:
		const unique_ptr<Matrix> GetReceptiveField(const Matrix& input, int filterSize) const {

			auto receptiveField = input.Copy();
			receptiveField->Initialise(filterSize, filterSize, false);

			for (auto row = 0; row < filterSize; row++) {
				for (auto col = 0; col < filterSize; col++) {
					(*receptiveField)(row, col) = input(row, col);
				}
			}

			return receptiveField;
		}
	};
}