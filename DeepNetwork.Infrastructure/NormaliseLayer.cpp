#include "NormaliseLayer.h"
#include <memory>
#include "Logging/logger.h"

void normalise_layer(matrix A) {
    auto logger = std::make_unique<Logger>();
    logger->LogLine("Normalising final layer");
    logger->LogDoubleArray(A.values, A.rows);

    double sum = 0;

    for (auto i = 0; i < A.rows; i++) {
        sum += A.values[i];
    }

    for (auto i = 0; i < A.rows; i++) {
        A.values[i] = A.values[i] / sum;
    }

    logger->LogLine("Final layer after normalisation: ");
    logger->LogDoubleArray(A.values, A.rows);
}