#pragma once
#include <memory>
#include "PyNetwork.h"
#include "PyNet.Models/Matrix.h"
#include "Headers.h"

using namespace std;

namespace PyNet::Infrastructure {

	class EXPORT NetworkRunner {

		shared_ptr<PyNetwork> _pyNetwork;

		NetworkRunner(shared_ptr<PyNetwork> pyNetwork) : _pyNetwork{ pyNetwork } {}
	public:

		static auto factory(shared_ptr<PyNetwork> pyNetwork) {
			return new NetworkRunner{ pyNetwork };
		}

		shared_ptr<Matrix> Run(double* inputLayer);
	};
}
