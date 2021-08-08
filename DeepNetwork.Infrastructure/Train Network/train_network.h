#pragma once

#include "../PyNetwork.h"

extern "C" {

	double train_network(PyNetwork* network, Matrix* expectedLayer);

}