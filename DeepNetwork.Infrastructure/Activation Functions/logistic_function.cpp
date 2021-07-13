#include "logistic_function.h"
#include "math.h"
#include "../Logging/logger.h"

void apply_logistic(matrix matrix) {

    auto logger = new Logger();
    logger->LogLine("Applying logistic activation function.");
    for (int i = 0; i < matrix.rows; i++) {
        float* mi = &matrix.values[i];
        *mi = 1 / (1 + exp(-*mi));
    }

    delete logger;
}

double calculate_logistic_derivative(double input) {
    return exp(input) / (1 + exp(input) * (1 + exp(input)));
}
