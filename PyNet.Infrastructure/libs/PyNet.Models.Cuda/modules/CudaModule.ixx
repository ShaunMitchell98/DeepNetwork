export module PyNet.Models.Cuda:CudaModule;

import PyNet.DI;
import :CudaMatrix;
import :CudaVector;

using namespace PyNet::DI;

export namespace PyNet::Models::Cuda {

	class __declspec(dllexport) CudaModule : public Module {

	public:
		void Load(ContextBuilder& builder) override {

			builder.RegisterType<CudaMatrix>(InstanceMode::Unique);
			builder.RegisterType<CudaVector>(InstanceMode::Unique);
		}
	};
}
