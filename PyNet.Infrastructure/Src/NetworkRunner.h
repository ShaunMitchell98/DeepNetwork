#pragma once
#include <memory>
#include "PyNetwork.h"
#include "LayerPropagator.h"
#include "LayerNormaliser.h"
#include "DropoutRunner.h"
#include "Headers.h"

using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkRunner {

		shared_ptr<PyNetwork> _pyNetwork;
		shared_ptr<LayerPropagator> _layerPropagator;
		shared_ptr<LayerNormaliser> _layerNormaliser;
		shared_ptr<DropoutRunner> _dropoutRunner;

		NetworkRunner(shared_ptr<PyNetwork> pyNetwork, shared_ptr<LayerPropagator> layerPropagator,
			shared_ptr<LayerNormaliser> layerNormaliser, shared_ptr<DropoutRunner> dropoutRunner) : _pyNetwork{ pyNetwork },
			_layerPropagator{ layerPropagator }, _layerNormaliser{ layerNormaliser }, _dropoutRunner{ dropoutRunner } {}
	public:

		static auto factory(shared_ptr<PyNetwork> pyNetwork, shared_ptr<LayerPropagator> layerPropagator,
			shared_ptr<LayerNormaliser> layerNormaliser, shared_ptr<DropoutRunner> dropoutRunner) {
			return new NetworkRunner{ pyNetwork, layerPropagator, layerNormaliser, dropoutRunner };
		}

		double* Run(double* inputLayer);
	};
}
