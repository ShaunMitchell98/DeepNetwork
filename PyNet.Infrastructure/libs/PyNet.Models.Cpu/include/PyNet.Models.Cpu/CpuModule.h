#pragma once
#include "PyNet.DI/ContextBuilder.h"
#include "PyNet.DI/Module.h"
#include "CpuMatrix.h"
#include "CpuVector.h"

using namespace PyNet::DI;

namespace PyNet::Models::Cpu {

	class CpuModule : public Module {

	public:
		void Load(ContextBuilder& builder) override {

			builder
				.RegisterType<CpuMatrix>(InstanceMode::Unique)
				.AsSelf()
				.As<Matrix>();

			builder
				.RegisterType<CpuVector>(InstanceMode::Unique)
				.AsSelf()
				.As<Vector>();
		}
	};
}