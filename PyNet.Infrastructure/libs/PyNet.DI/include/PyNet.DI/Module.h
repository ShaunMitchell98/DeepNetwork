#pragma once

#include "ContextBuilder.h"

namespace PyNet::DI {

	class Module {
	public:
		virtual void Load(ContextBuilder& builder) = 0;
	};
}