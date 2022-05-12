#pragma once
#include "PyNet.Models/ILogger.h"
#include <memory>

using namespace std;
using namespace PyNet::Models;

class LayerNormaliser {
private:
    shared_ptr<ILogger> _logger;
public:

    static auto factory(shared_ptr<ILogger> logger) {
        return new LayerNormaliser{ logger };
    }

    LayerNormaliser(shared_ptr<ILogger> logger) : _logger(logger) {}

	void NormaliseLayer(Matrix& mat) {

        _logger->LogLine("Normalising final layer");
        _logger->LogMatrix(mat);

        auto sum = 0.0;

        for (auto& i : mat) {
            sum += i;
        }

        for (auto& i : mat) {
            i = i / sum;
        }

        _logger->LogLine("Final layer after normalisation: ");
        _logger->LogMatrix(mat);
	}
};