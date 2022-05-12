#pragma once

#include <memory>
#include "Layers/TrainableLayer.h"

using namespace PyNet::Models;
using namespace PyNet::Infrastructure::Layers;
using namespace std;

class TrainingAlgorithm
{
public:
	virtual void UpdateWeights(vector<TrainableLayer*> layers, double learningRate, bool reverse = false) const = 0;
	virtual ~TrainingAlgorithm() = default;
};

