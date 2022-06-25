#pragma once

#include "ContextBuilder.h"

namespace PyNet::DI {

	class Module {
	public:
		virtual void Load(const ContextBuilder& builder) const = 0;
	};
}