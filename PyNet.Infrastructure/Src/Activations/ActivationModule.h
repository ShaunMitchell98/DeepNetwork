#pragma once

#include "PyNet.DI/Module.h"
#include "Logistic.h"
#include "Relu.h"

using namespace PyNet::DI;

namespace PyNet::Infrastructure::Activations {

	class ActivationModule : public Module {

	public:

		void Load(const ContextBuilder& builder) const override {

			builder.RegisterType<Logistic>()
				->AsSelf();

			builder.RegisterType<Relu>()
				->AsSelf();
		}
	};
}
