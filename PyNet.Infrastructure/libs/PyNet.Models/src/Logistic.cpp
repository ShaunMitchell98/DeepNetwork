#include "math.h"
#include "Logistic.h"

namespace ActivationFunctions {

    void Logistic::Apply(std::vector<double>& values) {

        for (int i = 0; i < values.size(); i++) {
            double* mi = &values[i];
            *mi = 1 / (1 + exp(-*mi));
        }
    }

    void Logistic::CalculateDerivative(PyNet::Models::Matrix* input, PyNet::Models::Matrix* output) {

        if (input->Rows != output->Rows) {
            throw "Invalid output vector provided.";
        }

        for (auto i = 0; i < input->Rows; i++) {
            for (auto j = 0; j < input->Cols; j++) {
                output->SetValue(i, j, exp(input->GetValue(i, j)) / ((1 + exp(input->GetValue(i, j)) * (1 + exp(input->GetValue(i, j))))));
            }
        }
    }
}