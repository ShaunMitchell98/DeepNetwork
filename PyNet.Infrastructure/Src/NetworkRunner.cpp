#include "NetworkRunner.h"
#include "Layers/InputLayer.h"

namespace PyNet::Infrastructure 
{    
	shared_ptr<Matrix> NetworkRunner::Run(shared_ptr<Matrix> input) 
	{
		auto inputLayer = static_cast<InputLayer*>(_pyNetwork->Layers.front().get());
		inputLayer->SetInput(input);

		shared_ptr<Matrix> output = input;

		for (const auto& layer : _pyNetwork->Layers) 
		{
			_dropoutRunner->ApplyDropout(*output, layer->DropoutRate);
			output = layer->Apply(move(output));
		}

		output = _softMax->Apply(output);

		return output;
	}
}
	