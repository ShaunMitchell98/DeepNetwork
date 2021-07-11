#include "logistic_function.h"
#include "math.h"
#include "../Logging/log.h"

void apply_logistic(matrix matrix) {

    log_line("Applying logistic activation function.");
    for (int i = 0; i < matrix.rows; i++) {
        float* mi = &matrix.values[i];
        *mi = 1 / (1 + exp(-*mi));
    }
}

float calculate_logistic_derivative(float input) {
    return exp(input) / (1 + exp(input) * (1 + exp(input)));
}
