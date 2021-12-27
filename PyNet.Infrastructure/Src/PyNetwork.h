#pragma once

#include "PyNet.Models/Matrix.h"
#include <memory>
#include <vector>
#include "PyNet.Models/Vector.h"
#include "NetworkTrainer.h"
#include "PyNet.DI/Context.h"
#include "LayerPropagator.h"
#include "PyNet.Models/ILogger.h"

namespace PyNet::Infrastructure {

	class PyNetwork
	{
	private:
		std::shared_ptr<LayerPropagator> _layerPropagator;
		std::shared_ptr<ILogger> _logger;
		std::shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
		std::shared_ptr<NetworkTrainer> _networkTrainer;
		std::shared_ptr<Settings> _settings;
		std::shared_ptr<PyNet::DI::Context> _context;
		std::shared_ptr<Loss> _loss;
		std::vector<double> _losses = std::vector<double>();
		std::vector<std::unique_ptr<PyNet::Models::Vector>> _layers = std::vector<std::unique_ptr<PyNet::Models::Vector>>();
		std::vector<std::unique_ptr<PyNet::Models::Vector>> _biases = std::vector<std::unique_ptr<PyNet::Models::Vector>>();
		std::vector<std::unique_ptr<PyNet::Models::Matrix>> _weights = std::vector<std::unique_ptr<PyNet::Models::Matrix>>();

	public:

		static auto factory(std::shared_ptr<ILogger> logger, std::shared_ptr<LayerPropagator> layerPropagator, std::shared_ptr<PyNet::DI::Context> context,
			std::shared_ptr<AdjustmentCalculator> adjustmentCalculator,
			std::shared_ptr<NetworkTrainer> networkTrainer, std::shared_ptr<Settings> settings, std::shared_ptr<Loss> loss) {
			return new PyNetwork{ logger, layerPropagator, context, adjustmentCalculator, networkTrainer, settings, loss };
		}

		PyNetwork(std::shared_ptr<ILogger> logger, std::shared_ptr<LayerPropagator> layerPropagator, std::shared_ptr<PyNet::DI::Context> context,
			std::shared_ptr<AdjustmentCalculator> adjustmentCalculator, std::shared_ptr<NetworkTrainer> networkTrainer, std::shared_ptr<Settings> settings, std::shared_ptr<Loss> loss) :
			_logger{ logger }, _layerPropagator{ layerPropagator }, _context{ context }, _adjustmentCalculator{ adjustmentCalculator },
			_networkTrainer{ networkTrainer }, _settings{ settings }, _loss{ loss } {}


		__declspec(dllexport) void AddInitialLayer(int rows);
		_declspec(dllexport) void AddLayer(int);
		_declspec(dllexport) double* Run(double* input_layer);
		_declspec(dllexport) double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate);
	};
}
