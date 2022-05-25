#pragma once

#include "PyNet.DI/Module.h"
#include "Layers/FakeLayer.h"

using namespace PyNet::Infrastructure::Layers;
using namespace PyNet::Infrastructure::Tests::Layers;

using namespace PyNet::DI;

namespace PyNet::Infrastructure::Tests 
{
	class IntegrationTestsModule : public Module
	{
	public:
		void Load(const ContextBuilder& builder) const override
		{
			builder.RegisterType<FakeLayer>()
				->AsSelf();
		}
	};
}