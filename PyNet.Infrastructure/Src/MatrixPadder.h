#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace std;

using namespace PyNet::Models;

namespace PyNet::Infrastructure 
{
	class MatrixPadder {
	public:

		const unique_ptr<Matrix> PadMatrix(const Matrix& input, int filterSize) const 
		{
			auto padding = (filterSize - 1) / 2;

			auto paddedMatrix = input.Copy();
			paddedMatrix->Initialise(input.GetRows() + static_cast<size_t>(padding), input.GetCols() + static_cast<size_t>(padding), false);

			for (size_t row = padding; row < (paddedMatrix->GetRows() - padding); row++)
			{
				for (size_t col = padding; col < (paddedMatrix->GetCols() - padding); col++) 
				{
					(*paddedMatrix)(row, col) = input(row - 1, col - 1);
				}
			}

			return paddedMatrix;
		}
	};
}