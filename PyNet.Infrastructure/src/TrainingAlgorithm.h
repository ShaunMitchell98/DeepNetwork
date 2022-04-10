#pragma once

#include <memory>
#include "PyNet.Models/Vector.h"

using namespace PyNet::Models;
using namespace std;

class TrainingAlgorithm
{
public:
	virtual void UpdateWeights(vector<unique_ptr<Matrix>>& weightMatrices, vector<unique_ptr<Vector>>& biases, double learningRate, bool reverse = false) = 0;
};

