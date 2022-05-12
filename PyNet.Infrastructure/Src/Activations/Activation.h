#pragma once

#include "../Layers/Layer.h"

using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure::Activations {

	enum class ActivationFunctionType {
		Logistic,
		Relu
	};

	class Activation : public Layer {
	};
}