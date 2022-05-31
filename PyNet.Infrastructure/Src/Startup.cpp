#include <memory>
#include "Startup.h"
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
using namespace std;
using namespace PyNet::Infrastructure::Activations;

namespace PyNet::Infrastructure {

	void Startup::RegisterServices(const ContextBuilder& builder, shared_ptr<Settings> settings) const {

#ifdef CUDA
		if (settings->CudaEnabled) {
			auto cudaModule = make_unique<CudaModule>();
			cudaModule->Load(builder);
		}
		else
		{
			auto cpuModule = make_unique<CpuModule>();
			cpuModule->Load(builder);
		}
#else
		auto cpuModule = make_unique<CpuModule>();
		cpuModule->Load(builder);
#endif

		auto infrastructureModule = make_unique<InfrastructureModule>(settings);
		infrastructureModule->Load(builder);

		auto layerModule = make_unique<Layers::LayerModule>();
		layerModule->Load(builder);

		auto activationModule = make_unique<ActivationModule>();
		activationModule->Load(builder);
	}
}