#pragma once
#include <vector>
#include <memory>
#include "PyNet.Models/Vector.h"
#include "PyNet.Models/Matrix.h"

using namespace std;
using namespace PyNet::Models;

struct PyNetwork
{
public:
	vector<double> Losses = vector<double>();
	vector<unique_ptr<Vector>> Layers = vector<unique_ptr<Vector>>();
	vector<unique_ptr<Vector>> Biases = vector<unique_ptr<Vector>>();
	vector<unique_ptr<Matrix>> Weights = vector<unique_ptr<Matrix>>();

	Vector& GetOutputLayer() {
		return *Layers[Layers.size() - 1];
	}

	double GetInputSize() const {
		return Layers[0]->GetRows();
	}

	static auto factory() {
		return new PyNetwork();
	}
};