#pragma once

#include "Context.h"
#include "Logger.h"

class UnitTest
{
public:

	di::ContextTmpl<PyNet::Infrastructure::Logger>* GetContext(bool log) {
		auto context = new di::ContextTmpl<PyNet::Infrastructure::Logger>();
		Settings* settings = new Settings();
		settings->LoggingEnabled = log;
		context->addInstance<Settings>(settings, true);
		return context;
	}
};

