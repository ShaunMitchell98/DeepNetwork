#include "NormaliseLayer.h"
#include <memory>

void normalise_layer(Models::Vector* A, ILogger* logger) {
    logger->LogLine("Normalising final layer");
    logger->LogDoubleArray(A->GetAddress(0), A->Rows);

    double sum = 0;

    for (auto i = 0; i < A->Rows; i++) {
        sum += A->GetValue(i);
    }

    for (auto i = 0; i < A->Rows; i++) {
        A->SetValue(i, A->GetValue(i) / sum);
    }

    logger->LogLine("Final layer after normalisation: ");
    logger->LogDoubleArray(A->GetAddress(0), A->Rows);
}