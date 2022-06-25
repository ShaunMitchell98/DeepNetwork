#pragma once
#include "PyNet.DI/ContextBuilder.h"
#include "PyNet.DI/Module.h"
#include "CpuMatrix.h"

using namespace PyNet::DI;

namespace PyNet::Models::Cpu {

	class CpuModule : public Module {

	public:
		void Load(const ContextBuilder& builder) const override {

			(*builder
				.RegisterType<CpuMatrix>(InstanceMode::Unique))
				.AsSelf()
				.As<Matrix>();
		}
	};
}