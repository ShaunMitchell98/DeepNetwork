#pragma once
#include <memory>
#include "PyNet.DI/Module.h"
#include "PyNet.DI/ContextBuilder.h"
#include "PyNet.DI/ItemRegistrar.h"
#include "CudaMatrix.h"
#include "CudaVector.h"

using namespace PyNet::DI;

namespace PyNet::Models::Cuda {

	class __declspec(dllexport) CudaModule : public Module {

	public:
		void Load(ContextBuilder& builder) override {

			(*builder
				.RegisterType<CudaMatrix>(InstanceMode::Unique))
				.AsSelf()
				.As<Matrix>();


			(*builder
				.RegisterType<CudaVector>(InstanceMode::Unique))
				.AsSelf()
				.As<Vector>();
		}
	};
}
