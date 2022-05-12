#pragma once
#include <memory>
#include "InfrastructureModule.h"
#include "Layers/LayerModule.h"
#include "Activations/ActivationModule.h"
#include "PyNet.Models.Cpu/CpuModule.h"
#include "Headers.h"

#ifdef CUDA
#include "PyNet.Models.Cuda/CudaModule.h"
using namespace PyNet::Models::Cuda;
#endif

using namespace PyNet::Models::Cpu;
using namespace PyNet::DI;
using namespace std;

namespace PyNet::Infrastructure {
	shared_ptr<Context> GetContext(shared_ptr<Settings> settings, bool cudaEnabled) {

		auto builder = make_unique<ContextBuilder>();

		if (cudaEnabled) {
			#ifdef CUDA
			auto cudaModule = make_unique<CudaModule>();
			cudaModule->Load(*builder);
			#endif
		}
		else
		{
			auto cpuModule = make_unique<CpuModule>();
			cpuModule->Load(*builder);
		}

		auto infrastructureModule = make_unique<InfrastructureModule>(settings);
		infrastructureModule->Load(*builder);

		auto layerModule = make_unique<Layers::LayerModule>();
		layerModule->Load(*builder);

		auto activationModule = make_unique<ActivationModule>();
		activationModule->Load(*builder);

		return builder->Build();
	}

}