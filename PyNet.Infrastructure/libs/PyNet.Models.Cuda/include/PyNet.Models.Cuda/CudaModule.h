#pragma once

#include "PyNet.DI/Module.h"
#include "PyNet.DI/Item.h"
#include "CudaMatrix.h"
#include "CudaVector.h"

namespace PyNet::Models::Cuda {

	class CudaModule : public PyNet::DI::Module {

	public:
		void Load(PyNet::DI::ContextBuilder* builder) override {

			builder->AddClass<CudaMatrix>(PyNet::DI::InstanceMode::Unique);
			builder->AddClass<CudaVector>(PyNet::DI::InstanceMode::Unique);
		}
	};
}
