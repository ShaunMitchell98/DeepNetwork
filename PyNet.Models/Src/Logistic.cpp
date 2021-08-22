#include "Logistic.h"
#include "math.h"

namespace ActivationFunctions {

    void Logistic::Apply(std::vector<double>& values) {

        for (int i = 0; i < values.size(); i++) {
            double* mi = &values[i];
            *mi = 1 / (1 + exp(-*mi));
        }
    }

    double Logistic::CalculateDerivative(double input) {
        return exp(input) / ((1 + exp(input)) * (1 + exp(input)));
    }
}

