module;
#include <memory>
export module PyNet.Infrastructure:NetworkRunner;

import :PyNetwork;
import :LayerPropagator;
import :LayerNormaliser;

using namespace std;

namespace PyNet::Infrastructure {

	class NetworkRunner {

		shared_ptr<PyNetwork> _pyNetwork;
		shared_ptr<LayerPropagator> _layerPropagator;
		shared_ptr<LayerNormaliser> _layerNormaliser;

		NetworkRunner(shared_ptr<PyNetwork> pyNetwork, shared_ptr<LayerPropagator> layerPropagator,
			shared_ptr<LayerNormaliser> layerNormaliser) : _pyNetwork{ pyNetwork },
			_layerPropagator{ layerPropagator }, _layerNormaliser{ layerNormaliser } {}
	public:

		static auto factory(shared_ptr<PyNetwork> pyNetwork, shared_ptr<LayerPropagator> layerPropagator,
			shared_ptr<LayerNormaliser> layerNormaliser) {
			return new NetworkRunner{ pyNetwork, layerPropagator, layerNormaliser };
		}

		double* Run(double* inputLayer) {

			_pyNetwork->Layers[0]->Set(_pyNetwork->GetInputSize(), inputLayer);

			for (size_t i = 0; i < _pyNetwork->Weights.size(); i++) {
				_layerPropagator->PropagateLayer(*_pyNetwork->Weights[i], *_pyNetwork->Layers[i], *_pyNetwork->Biases[i], *_pyNetwork->Layers[i + 1]);
			}

			_layerNormaliser->NormaliseLayer(_pyNetwork->GetLastLayer());

			return _pyNetwork->GetLastLayer().GetAddress(0);
		}
	};
}

