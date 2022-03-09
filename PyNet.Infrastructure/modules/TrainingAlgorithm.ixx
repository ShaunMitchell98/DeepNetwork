module;
#include <memory>
#include <vector>
export module PyNet.Infrastructure:TrainingAlgorithm;

import PyNet.Models;

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {
	export class TrainingAlgorithm
	{
	public:
		virtual void UpdateWeights(vector<unique_ptr<Matrix>>& weightMatrices, vector<unique_ptr<Vector>>& biases, double learningRate, bool reverse = false) = 0;
	};

}

