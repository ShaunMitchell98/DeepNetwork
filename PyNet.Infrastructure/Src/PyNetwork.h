#pragma once
#include <vector>
#include <memory>
#include "Layers/Layer.h"

using namespace std;
using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure {
	struct PyNetwork
	{
	public:
		vector<unique_ptr<Layer>> Layers = vector<unique_ptr<Layer>>();

		static auto factory() {
			return new PyNetwork();
		}
	};
}

