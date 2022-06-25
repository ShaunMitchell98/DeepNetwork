#pragma once

#include "Startup.h"
#include "Settings.h"
#include "IntegrationTestsModule.h"
#include <memory>

namespace PyNet::Infrastructure::Tests 
{
	class ContainerFixture
	{
	public:
		static shared_ptr<Context> Initialise() 
		{
			auto settings = make_shared<Settings>();
			auto startup = make_unique<Startup>();
			
			auto contextBuilder = make_unique<ContextBuilder>();
			startup->RegisterServices(*contextBuilder, settings);

			auto testsModule = make_unique<IntegrationTestsModule>();
			testsModule->Load(*contextBuilder);

			return contextBuilder->Build();
		}
	};
}