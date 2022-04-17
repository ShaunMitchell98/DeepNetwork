#pragma once
#include <memory>
#include "PyNetwork.h"
#include "LayerPropagator.h"
#include "LayerNormaliser.h"
#include "Headers.h"

using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkRunner {

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

		double* Run(double* inputLayer);
	};
}
