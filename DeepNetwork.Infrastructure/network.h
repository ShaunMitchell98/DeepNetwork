#pragma once

#include "matrix.h"

struct network {
	matrix* layers;
	matrix* weights;
	int layerCount; 
	int weightMatrixCount;
};