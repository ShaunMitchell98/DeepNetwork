module;
#include <memory>
export module PyNet.Infrastructure:LayerPropagator;

import PyNet.Models;

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

    class __declspec(dllexport) LayerPropagator {
    private:
        shared_ptr<ILogger> _logger;
    public:

        auto static factory(shared_ptr<ILogger> logger) {
            return new LayerPropagator(logger);
        }

        LayerPropagator(shared_ptr<ILogger> logger) : _logger(logger) {}

        void PropagateLayer(const Matrix& weights, const Vector& inputLayer, const Vector& bias, Vector& outputLayer) const {

            _logger->LogLine("Forward propagating layer.");

            _logger->LogLine("Forward propagation input: ");
            _logger->LogMessage(inputLayer.ToString());

            outputLayer = *(weights * inputLayer);

            _logger->LogLine("Forward propagation output: ");
            _logger->LogMessage(outputLayer.ToString());

            outputLayer += bias;

            _logger->LogLine("Output after adding biases: ");
            _logger->LogMessage(outputLayer.ToString());

            outputLayer.ApplyActivation();
            _logger->LogLine("Output after activation");
            _logger->LogMessage(outputLayer.ToString());
        }
    };
}