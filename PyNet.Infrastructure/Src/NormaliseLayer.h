#pragma once

#include "PyNet.Models/Vector.h"
#include "PyNet.Models/ILogger.h"

using namespace PyNet::Models;

void normalise_layer(Vector& A, ILogger& logger) {
    logger.LogLine("Normalising final layer");
    logger.LogMessage(A.ToString());

    auto sum = 0.0;

    for (auto i = 0; i < A.GetRows(); i++) {
        sum += A[i];
    }

    for (auto i = 0; i < A.GetRows(); i++) {
        A[i] = A[i] / sum;
    }

    logger.LogLine("Final layer after normalisation: ");
    logger.LogMessage(A.ToString());
}