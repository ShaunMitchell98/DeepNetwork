#pragma once

#include "Headers.h"

namespace PyNet::Infrastructure {
	extern "C" {

		EXPORT int PyNetwork_Load(void* input, const char* filePath);

		EXPORT void PyNetwork_Save(void* input, const char* filePath);
	}
}