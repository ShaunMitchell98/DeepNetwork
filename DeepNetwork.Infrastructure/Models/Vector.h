#pragma once

#include "Matrix.h"
#include "../Activation Functions/Activation.h"
#include <memory>
#include "../Activation Functions/Logistic.h"

using namespace ActivationFunctions;

namespace Models {

	class Vector : public Matrix
	{
	private:
		std::unique_ptr<Activation> _activation;
	public:
		Vector(int rows, ActivationFunctionType activationFunctionType) : Vector(rows) {
			if (activationFunctionType == ActivationFunctionType::Logistic) {
				_activation = std::make_unique<Logistic>();
			}
		}
		Vector(int rows, double* values, ActivationFunctionType activationFunctionType) : Matrix(rows, 1, values) {
			if (activationFunctionType == ActivationFunctionType::Logistic) {
				_activation = std::make_unique<Logistic>();
			}
		}

		Vector(int rows) : Matrix(rows, 1) {}

		double GetValue(int row);
		double* GetAddress(int row);
		void SetValue(int row, double value);
		void ApplyActivation();
		double CalculateActivationDerivative(double input);
	};
}

