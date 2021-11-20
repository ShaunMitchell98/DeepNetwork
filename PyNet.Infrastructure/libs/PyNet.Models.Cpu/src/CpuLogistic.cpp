#include "math.h"
#include "CpuLogistic.h"

namespace PyNet::Models::Cpu {

    void CpuLogistic::Apply(PyNet::Models::Matrix& input) {

        for (int i = 0; i < input.GetRows(); i++) {
            for (int j = 0; j < input.GetCols(); j++) {
                input.SetValue(i, j, 1 / (1 + exp(-input.GetValue(i, j))));
            }
        }
    }

    void CpuLogistic::CalculateDerivative(PyNet::Models::Matrix& input, PyNet::Models::Matrix& output) {

        for (auto i = 0; i < input.GetRows(); i++) {
            for (auto j = 0; j < input.GetCols(); j++) {
                output.SetValue(i, j, exp(input.GetValue(i, j)) / ((1 + exp(input.GetValue(i, j)) * (1 + exp(input.GetValue(i, j))))));
            }
        }
    }
}