#include "DropoutRunner.h"

namespace PyNet::Infrastructure {

	void DropoutRunner::ApplyDropout(Matrix& input, double rate) const {

		if (rate == 1) 
		{
			return;
		}

		unique_ptr<Matrix> droppedMatrix;
		if (_settings->RunMode == RunMode::Training)
		{
			droppedMatrix = _bernoulliGenerator->GetBernoulliVector(input, rate);
		}
		else
		{
			droppedMatrix = input.Copy();

			for (auto& m : *droppedMatrix)
			{
				m = rate;
			}
		}

		input = *(input ^ *droppedMatrix);
	}
}