#pragma once

#include "../matrix.h"
#include "../Logging/logger.h"

void apply_logistic(Matrix* matrix, Logger* logger);

double calculate_logistic_derivative(double input);