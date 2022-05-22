#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	class ReceptiveFieldProvider {
	public:

		static auto factory() {
			return new ReceptiveFieldProvider();
		}

		const unique_ptr<Matrix> GetReceptiveField(const Matrix& input, int filterSize, int rStart, int cStart) const {

			auto receptiveField = input.Copy();
			receptiveField->Initialise(filterSize, filterSize, false);

			for (size_t row = 1; row <= filterSize; row++) {
				for (size_t col = 1; col <= filterSize; col++) {
					(*receptiveField)(row, col) = input(row + rStart, col+cStart);
				}
			}

			return receptiveField;
		}
	};
}