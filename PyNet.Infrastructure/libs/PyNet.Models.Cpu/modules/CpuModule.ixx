export module PyNet.Models.Cpu:CpuModule;

import PyNet.DI;
import :CpuMatrix;
import :CpuVector;

using namespace PyNet::DI;

export namespace PyNet::Models::Cpu {

	class CpuModule : public Module {

	public:
		void Load(ContextBuilder& builder) override {

			builder.RegisterType<CpuMatrix>(InstanceMode::Unique);
			builder.RegisterType<CpuVector>(InstanceMode::Unique);
		}
	};
}