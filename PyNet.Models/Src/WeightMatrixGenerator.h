#pragma once

#include <cstdlib> 

void generate_random_weights(double* address, int count) {

	for (int i = 0; i < count; i++) {
		address[i] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX) * 100);
	}
}