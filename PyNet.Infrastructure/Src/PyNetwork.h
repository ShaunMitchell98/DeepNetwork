#pragma once

#include "PyNet.Models/Matrix.h"
#include <memory>
#include <vector>
#include "PyNet.Models/Vector.h"
#include "TrainingAlgorithm.h"
#include "AdjustmentCalculator.h"
#include "PyNet.DI/Context.h"
#include "PyNet.Models/Loss.h"
#include "LayerPropagator.h"
#include "GradientCalculator.h"
#include "PyNet.Models/ILogger.h"
#include "VariableLearningSettings.h"

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	class PyNetwork
	{
	private:
		shared_ptr<LayerPropagator> _layerPropagator;
		shared_ptr<ILogger> _logger;
		shared_ptr<AdjustmentCalculator> _adjustmentCalculator;
		shared_ptr<TrainingAlgorithm> _trainingAlgorithm;
		shared_ptr<Settings> _settings;
		shared_ptr<PyNet::DI::Context> _context;
		shared_ptr<GradientCalculator> _gradientCalculator;
		shared_ptr<Loss> _loss;
		VariableLearningSettings* _vlSettings = nullptr;
		vector<double> _losses = vector<double>();
		vector<unique_ptr<Vector>> _layers = vector<unique_ptr<Vector>>();
		vector<unique_ptr<Vector>> _biases = vector<unique_ptr<Vector>>();
		vector<unique_ptr<Matrix>> _weights = vector<unique_ptr<Matrix>>();

	public:

		static auto factory(shared_ptr<ILogger> logger, shared_ptr<LayerPropagator> layerPropagator, shared_ptr<PyNet::DI::Context> context,
			shared_ptr<GradientCalculator> gradientCalculator, shared_ptr<AdjustmentCalculator> adjustmentCalculator,
			shared_ptr<TrainingAlgorithm> trainingAlgorithm, shared_ptr<Settings> settings, shared_ptr<Loss> loss) {
			return new PyNetwork{ logger, layerPropagator, context, gradientCalculator, adjustmentCalculator, trainingAlgorithm, settings, loss };
		}

		PyNetwork(shared_ptr<ILogger> logger, shared_ptr<LayerPropagator> layerPropagator, shared_ptr<PyNet::DI::Context> context, shared_ptr<GradientCalculator> gradientCalculator,
			shared_ptr<AdjustmentCalculator> adjustmentCalculator, shared_ptr<TrainingAlgorithm> trainingAlgorithm, shared_ptr<Settings> settings, shared_ptr<Loss> loss) :
			_logger{ logger }, _layerPropagator{ layerPropagator }, _context{ context }, _gradientCalculator{ gradientCalculator }, _adjustmentCalculator{ adjustmentCalculator },
			_trainingAlgorithm{ trainingAlgorithm }, _settings{ settings }, _loss{ loss } {}

	    int Load(const char* filePath);
		void AddLayer(int rows);
		double* Run(double* input_layer);
		double* Train(double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double baseLearningRate,
			double momentum, int epochs);
		void SetVLSettings(VariableLearningSettings* vlSettings);
		void Save(const char* filePath);
	};
}
