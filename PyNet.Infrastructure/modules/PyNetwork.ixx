module;
#include <vector>
#include <memory>
export module PyNet.Infrastructure:PyNetwork;

import PyNet.Models;

using namespace std;
using namespace PyNet::Models;

export struct PyNetwork
{
public:
	vector<double> Losses = vector<double>();
	vector<unique_ptr<Vector>> Layers = vector<unique_ptr<Vector>>();
	vector<unique_ptr<Vector>> Biases = vector<unique_ptr<Vector>>();
	vector<unique_ptr<Matrix>> Weights = vector<unique_ptr<Matrix>>();

	Vector& GetLastLayer() {
		return *Layers[Layers.size() - 1];
	}

	double GetInputSize() const {
		return Layers[0]->GetRows();
	}

	__declspec(dllexport) static auto factory() {
		return new PyNetwork();
	}
};