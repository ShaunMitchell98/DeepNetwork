#pragma once

#include "PyNet.Models/Context.h"
#include "Logger.h"

class UnitTest
{
public:

	di::Context* GetContext() {
		auto context = new di::ContextTmpl<PyNet::Infrastructure::Logger>();
		Settings* settings = new Settings();
		settings->LoggingEnabled = false;
		context->addInstance<Settings>(settings, true);
		return context;
	}
};

