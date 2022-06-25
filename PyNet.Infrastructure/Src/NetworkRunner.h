#pragma once
#include <memory>
#include "PyNetwork.h"
#include "PyNet.Models/Matrix.h"
#include "Layers/SoftmaxLayer.h"
#include "DropoutRunner.h"
#include "Headers.h"

using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkRunner {

		shared_ptr<PyNetwork> _pyNetwork;
		unique_ptr<SoftmaxLayer> _softMax;
		unique_ptr<DropoutRunner> _dropoutRunner;

		NetworkRunner(shared_ptr<PyNetwork> pyNetwork, unique_ptr<SoftmaxLayer> softmaxLayer, unique_ptr<DropoutRunner> dropoutRunner) : _pyNetwork{ pyNetwork },
			_softMax{ move(softmaxLayer) }, _dropoutRunner{ move(dropoutRunner) } {}
	public:

		static auto factory(shared_ptr<PyNetwork> pyNetwork, unique_ptr<SoftmaxLayer> softmax, unique_ptr<DropoutRunner> dropoutRunner) {
			return new NetworkRunner{ pyNetwork, move(softmax), move(dropoutRunner) };
		}

		shared_ptr<Matrix> Run(shared_ptr<Matrix> input);
	};
}
