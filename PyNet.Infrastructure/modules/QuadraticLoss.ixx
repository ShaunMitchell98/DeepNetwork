module;
#include <memory>
export module PyNet.Infrastructure:QuadraticLoss;

import PyNet.Models;

using namespace PyNet::Models;
using namespace std;

class __declspec(dllexport) QuadraticLoss : public Loss {
public:

	static auto factory() {
		return new QuadraticLoss();
	}

	typedef Loss base;

	double CalculateLoss(Vector& expected, Vector& actual) override {
		auto difference = expected - actual;
		return 0.5 * (*difference | *difference);
	}

	unique_ptr<Vector> CalculateDerivative(Vector& expected, Vector& actual) override {
		return actual - expected;
	}
};