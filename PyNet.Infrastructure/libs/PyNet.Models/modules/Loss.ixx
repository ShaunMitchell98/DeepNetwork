module;
#include <memory>
export module PyNet.Models:Loss;

using namespace std;

import :Vector;

export namespace PyNet::Models {

	class Loss {
	public:
		virtual	double CalculateLoss(Vector& expected, Vector& actual) = 0;
		virtual unique_ptr<Vector> CalculateDerivative(Vector& expected, Vector& actual) = 0;
	};
}
