#include "logistic_function.h"
#include "math.h"

void apply_logistic(Matrix* matrix, Logger* logger) {

    logger->LogLine("Applying logistic activation function.");
    for (int i = 0; i < matrix->Rows; i++) {
        double* mi = &matrix->Values[i];
        *mi = 1 / (1 + exp(-*mi));
    }
}

double calculate_logistic_derivative(double input) {
    return exp(input) / (1 + exp(input) * (1 + exp(input)));
}
