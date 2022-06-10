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

	void NormaliseLayer(Vector& v) {

        //_logger->LogLine("Normalising final layer");
        //_logger->LogMessage(v.ToString());

        auto sum = 0.0;

        for (auto i = 0; i < v.GetRows(); i++) {
            sum += v[i];
        }

        for (auto i = 0; i < v.GetRows(); i++) {
            v[i] = v[i] / sum;
        }

        //_logger->LogLine("Final layer after normalisation: ");
        //_logger->LogMessage(v.ToString());
	}
};