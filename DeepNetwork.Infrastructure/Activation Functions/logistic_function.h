#pragma once

#include "../matrix.h"
#include "../Logging/logger.h"

void apply_logistic(matrix matrix, Logger* logger);

double calculate_logistic_derivative(double input);