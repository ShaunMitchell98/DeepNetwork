#pragma once
#include <memory>
#include "Settings.h"
#include "PyNet.DI/Context.h"
#include "Headers.h"

using namespace PyNet::DI;

namespace PyNet::Infrastructure {
	shared_ptr<Context> EXPORT GetContext(shared_ptr<Settings> settings, bool cudaEnabled);
}