#pragma once
#include <memory>
#include "PyNet.DI/Module.h"
#include "PyNet.DI/ContextBuilder.h"
#include "PyNet.DI/ItemRegistrar.h"
#include "CudaMatrix.h"

using namespace PyNet::DI;

namespace PyNet::Models::Cuda {

	class __declspec(dllexport) CudaModule : public Module {

	public:
		void Load(const ContextBuilder& builder) const override {

			(*builder
				.RegisterType<CudaMatrix>(InstanceMode::Unique))
				.AsSelf()
				.As<Matrix>();
		}
	};
}
