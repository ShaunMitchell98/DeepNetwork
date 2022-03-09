module;
#include <memory>
export module PyNet.Infrastructure:Setup;

import :InfrastructureModule;
import PyNet.DI;
import PyNet.Models.Cuda;
import PyNet.Models.Cpu;

using namespace PyNet::Models::Cuda;
using namespace PyNet::Models::Cpu;
using namespace PyNet::DI;
using namespace std;

export namespace PyNet::Infrastructure {
	__declspec(dllexport) shared_ptr<Context> GetContext(bool cudaEnabled, bool log) {

		auto builder = make_unique<ContextBuilder>();

		if (cudaEnabled) {
			auto cudaModule = make_unique<CudaModule>();
			cudaModule->Load(*builder);
		}
		else
		{
			auto cpuModule = make_unique<CpuModule>();
			cpuModule->Load(*builder);
		}

		auto infrastructureModule = make_unique<InfrastructureModule>(log);
		infrastructureModule->Load(*builder);

		return builder->Build();
	}

}