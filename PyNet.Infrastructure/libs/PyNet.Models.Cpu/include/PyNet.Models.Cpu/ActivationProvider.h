#pragma once

#include "CpuLogistic.h"

namespace ActivationFunctions {
	auto factory() {
		return new CpuLogistic();
	}
}
