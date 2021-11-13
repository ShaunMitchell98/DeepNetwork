#pragma once

#include "CpuLogistic.h"

namespace PyNet::Models::Cpu {

	auto factory() {
		return new CpuLogistic();
	}
}
