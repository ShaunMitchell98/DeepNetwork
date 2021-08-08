#pragma once

#include "../Matrix.h"
#include "../Activation Functions/activation_function.h"

extern "C" {

	void forward_propagate_layer(Matrix* weights, Matrix* inputLayer, Matrix* outputLayer, activation_function activationFunction);

}