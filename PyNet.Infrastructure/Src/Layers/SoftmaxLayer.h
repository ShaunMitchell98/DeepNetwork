#pragma once
#include "Layer.h"
#include "PyNet.Models/Matrix.h"
#include <memory>

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure::Layers
{
    class SoftmaxLayer : public Activation
    {
        private:

        SoftmaxLayer(unique_ptr<Matrix> input, unique_ptr<Matrix> output) : Activation(move(input)) {}

        public:

        static auto factory(unique_ptr<Matrix> input, unique_ptr<Matrix> output)
        {
            return new SoftmaxLayer{ move(input), move(output) };
        }

        void Initialise(size_t rows, size_t cols)
        {
            Input->Initialise(rows, cols, false);
        }

        shared_ptr<Matrix> Apply(const shared_ptr<Matrix> input) override
        {
            Input = input;

            auto sum = 0.0;

            for (auto& i : *input)
            {
                sum += i;
            }

            for (auto row = 1; row <= input->GetRows(); row++) 
            {
                (*Input)(row, 1) = (*input)(row, 1) / sum;
            }

            return Input;
        }

        unique_ptr<Matrix> Derivative(const Matrix& dLoss_dOutput) const override
        {
            /*if (dLoss_dOutput.GetCols() != 1) 
            {
                throw "Expected a vector input";
            }

            auto dLoss_dInput = dLoss_dOutput.Copy();

            auto product = dLoss_dOutput | *Output;

            for (auto row = 1; row < dLoss_dOutput.GetRows(); row++) 
            {
                (*dLoss_dInput)(row, 1) = (*Output)(row, 1) * ((dLoss_dOutput)(row, 1) - product);
            }*/

            //return dLoss_dInput;

            auto dLoss_dInput = dLoss_dOutput.Copy();
            dLoss_dInput->Set(dLoss_dOutput.GetRows(), dLoss_dOutput.GetCols(), dLoss_dOutput.GetAddress(1, 1));
            return dLoss_dInput;
        }
    };
}