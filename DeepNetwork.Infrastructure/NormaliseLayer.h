#ifndef KERNEL_NORMALISE_LAYER
#define KERNEL_NORMALISE_LAYER

#include "Models/Vector.h"
#include "Logging/ILogger.h"

extern "C" {

	void normalise_layer(Models::Vector* A, ILogger* logger);

}

#endif