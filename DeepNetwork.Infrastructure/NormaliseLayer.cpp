#include "NormaliseLayer.h"
#include <memory>
#include "Logging/logger.h"

void normalise_layer(Matrix* A) {
    auto logger = std::make_unique<Logger>();
    logger->LogLine("Normalising final layer");
    logger->LogDoubleArray(A->Values.data(), A->Rows);

    double sum = 0;

    for (auto i = 0; i < A->Rows; i++) {
        sum += A->Values[i];
    }

    for (auto i = 0; i < A->Rows; i++) {
        A->Values[i] = A->Values[i] / sum;
    }

    logger->LogLine("Final layer after normalisation: ");
    logger->LogDoubleArray(A->Values.data(), A->Rows);
}