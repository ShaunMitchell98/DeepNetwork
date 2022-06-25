#pragma once

#include <memory>
#include "PyNet.Models/Matrix.h"

using namespace std;

using namespace PyNet::Models;

namespace PyNet::Infrastructure 
{
	class MatrixPadder {
	public:

		static auto factory() {
			return new MatrixPadder();
		}

		const shared_ptr<Matrix> PadMatrix(const Matrix& input, size_t filterSize) const 
		{
			size_t padding = (filterSize - 1) / 2;

			auto paddedMatrix = input.Copy();
			paddedMatrix->Initialise(input.GetRows() + (2 * padding), input.GetCols() + (2 * padding), false);

			for (size_t row = padding+1; row <= (paddedMatrix->GetRows() - padding); row++)
			{
				for (size_t col = padding+1; col <= (paddedMatrix->GetCols() - padding); col++) 
				{
					(*paddedMatrix)(row, col) = input(row - 1, col - 1);
				}
			}

			return paddedMatrix;
		}
	};
}