#pragma once
#include <memory>
#include "PyNetwork.h"
#include "PyNet.Models/Matrix.h"
#include "Layers/SoftmaxLayer.h"
#include "Headers.h"

using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkRunner {

		shared_ptr<PyNetwork> _pyNetwork;
		unique_ptr<SoftmaxLayer> _softMax;

		NetworkRunner(shared_ptr<PyNetwork> pyNetwork, unique_ptr<SoftmaxLayer> softmaxLayer) : _pyNetwork{ pyNetwork }, _softMax{ move(softmaxLayer) } {}
	public:

		static auto factory(shared_ptr<PyNetwork> pyNetwork, unique_ptr<SoftmaxLayer> softmax) {
			return new NetworkRunner{ pyNetwork, move(softmax) };
		}

		shared_ptr<Matrix> Run(shared_ptr<Matrix> input);
	};
}
