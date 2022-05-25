#pragma once
#include "Layer.h"
#include "PyNet.Models/Matrix.h"
#include <memory>

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure::Layers
{
    class SoftmaxLayer : public Layer
    {
        private:
        SoftmaxLayer(unique_ptr<Matrix> input) : Layer(move(input)) {}

        public:

        static auto factory(unique_ptr<Matrix> input)
        {
            return new SoftmaxLayer{ move(input) };
        }

        shared_ptr<Matrix> Apply(const shared_ptr<Matrix> input) override
        {
            Input = input;
            Output = input->Copy();

            auto sum = 0.0;

            for (auto& i : *input)
            {
                sum += exp(i);
            }

            for (auto row = 1; row < input->GetRows(); row++) 
            {
                (*Output)(row, 1) = exp((*input)(row, 1)) / sum;
            }

            return Output;
        }

        unique_ptr<Matrix> dLoss_dInput(const Matrix& dLoss_dOutput) const 
        {
            if (dLoss_dOutput.GetCols() != 1) 
            {
                throw "Expected a vector input";
            }

            auto dLoss_dInput = dLoss_dOutput.Copy();

            auto product = dLoss_dOutput | *Output;

            for (auto row = 1; row < dLoss_dOutput.GetRows(); row++) 
            {
                (*dLoss_dInput)(row, 1) = (*Output)(row, 1) * ((dLoss_dOutput)(row, 1) - product);
            }

            return dLoss_dInput;
        }
    };
}