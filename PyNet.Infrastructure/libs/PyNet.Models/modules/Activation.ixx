module;
#include <memory>
export module PyNet.Models:Activation;

import :Matrix;

using namespace std;

export namespace PyNet::Models {

	enum class ActivationFunctionType {
		Logistic
	};

	class Activation {
	protected:
	public:
		virtual void Apply(Matrix& input) = 0;
		virtual unique_ptr<Matrix> CalculateDerivative(const Matrix& input) = 0;
	};
}