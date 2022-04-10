#pragma once

#include "PyNet.DI/Module.h"
#include "PyNet.DI/Item.h"
#include "CpuMatrix.h"
#include "CpuVector.h"

namespace PyNet::Models::Cpu {

	class CpuModule : public PyNet::DI::Module {

	public:
		void Load(PyNet::DI::ContextBuilder* builder) override {

			builder->AddClass<CpuMatrix>(PyNet::DI::InstanceMode::Unique);
			builder->AddClass<CpuVector>(PyNet::DI::InstanceMode::Unique);
		}
	};
}
