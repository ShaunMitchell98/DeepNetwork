#include "DropoutRunner.h"

namespace PyNet::Infrastructure {

	void DropoutRunner::ApplyDropout(Vector& input) const {

		unique_ptr<Vector> droppedVector;
		if (_settings->RunMode == RunMode::Training) {
			droppedVector = _bernoulliGenerator->GetBernoulliVector(input);
		}
		else {
			droppedVector = input.CopyAsVector();
			droppedVector->SetValue(input.GetDropoutRate());
		}

		input = *(input ^ *droppedVector);
	}
}