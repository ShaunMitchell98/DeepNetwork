#pragma once

#include "Context.h"
#include "Logger.h"

class UnitTest
{
public:

	di::ContextTmpl<PyNet::Infrastructure::Logger>* GetContext(bool cudaEnabled, bool log) {
		auto context = new di::ContextTmpl<PyNet::Infrastructure::Logger>();
		Settings* settings = new Settings();
		settings->CudaEnabled = cudaEnabled;
		settings->LoggingEnabled = log;
		context->addInstance<Settings>(settings, true);
		return context;
	}
};

