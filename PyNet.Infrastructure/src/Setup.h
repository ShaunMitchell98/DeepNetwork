#pragma once

#include <PyNet.Models.Cuda/CudaModule.h>
#include <PyNet.Models.Cpu/CpuModule.h>
#include "InfrastructureModule.h"
#include <PyNet.DI/Context.h>
#include "Settings.h"

namespace PyNet::Infrastructure {

	inline std::shared_ptr<PyNet::DI::Context> GetContext(bool cudaEnabled, bool log) {

		auto builder = std::make_unique<PyNet::DI::ContextBuilder>();

		if (cudaEnabled) {
			auto cudaModule = std::make_unique<PyNet::Models::Cuda::CudaModule>();
			cudaModule->Load(builder.get());
		}
		else
		{
			auto cpuModule = std::make_unique<PyNet::Models::Cpu::CpuModule>();
			cpuModule->Load(builder.get());
		}

		auto infrastructureModule = std::make_unique<InfrastructureModule>(log);
		infrastructureModule->Load(builder.get());

		return builder->Build();
	}
}

