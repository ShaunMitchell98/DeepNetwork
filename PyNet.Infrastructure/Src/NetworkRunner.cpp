#include "NetworkRunner.h"
#include "Layers/InputLayer.h"

namespace PyNet::Infrastructure {
    
	shared_ptr<Matrix> NetworkRunner::Run(double* input) {

		auto inputLayer = dynamic_cast<InputLayer*>(_pyNetwork->Layers[0].get());
		inputLayer->SetInput(input);

		shared_ptr<Matrix> output;

		for (const auto& layer : _pyNetwork->Layers) {
			output = layer->Apply(move(output));
		}

		_layerNormaliser->NormaliseLayer(*output);

		return output;
	}
}
	