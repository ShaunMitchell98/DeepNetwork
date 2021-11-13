#pragma once

#include "PyNet.Models/Vector.h"
#include "PyNet.Models/ILogger.h"

void normalise_layer(PyNet::Models::Vector& A, ILogger& logger) {
    logger.LogLine("Normalising final layer");
    logger.LogMessage(A.ToString());

    double sum = 0;

    for (auto i = 0; i < A.GetRows(); i++) {
        sum += A.GetValue(i);
    }

    for (auto i = 0; i < A.GetRows(); i++) {
        A.SetValue(i, A.GetValue(i) / sum);
    }

    logger.LogLine("Final layer after normalisation: ");
    logger.LogMessage(A.ToString());
}