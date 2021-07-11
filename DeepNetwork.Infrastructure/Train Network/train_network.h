#pragma once

#include "../network.h"

extern "C" {
#define EXPORT_SYMBOL __declspec(dllexport)

	EXPORT_SYMBOL float train_network(network network, matrix expectedLayer);

#undef EXPORT_SYMBOL
}