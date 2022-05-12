#pragma once
#include <memory>
#include "PyNetwork.h"
#include "PyNet.Models/Matrix.h"
#include "LayerNormaliser.h"
#include "Headers.h"

using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkRunner {

		shared_ptr<PyNetwork> _pyNetwork;
		shared_ptr<LayerNormaliser> _layerNormaliser;

		NetworkRunner(shared_ptr<PyNetwork> pyNetwork, shared_ptr<LayerNormaliser> layerNormaliser) : _pyNetwork{ pyNetwork }, _layerNormaliser{ layerNormaliser } {}
	public:

		static auto factory(shared_ptr<PyNetwork> pyNetwork, shared_ptr<LayerNormaliser> layerNormaliser) {
			return new NetworkRunner{ pyNetwork, layerNormaliser};
		}

		unique_ptr<Matrix> Run(double* inputLayer);
	};
}
