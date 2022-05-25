#pragma once
#include <memory>
#include "Settings.h"
#include "PyNet.DI/ContextBuilder.h"
#include "Headers.h"

using namespace PyNet::DI;

namespace PyNet::Infrastructure 
{
	class EXPORT Startup 
	{
	public:
		void RegisterServices(const ContextBuilder& builder, shared_ptr<Settings> settings) const;
	};
}