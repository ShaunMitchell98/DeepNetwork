#include "CppUnitTest.h"
#include "LayerPropagator.h"
#include "Logger.h"
#include <memory>
#include "UnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PyNet::Infrastructure::Tests
{
	TEST_CLASS(LayerPropagatorTests), public UnitTest
	{
	public:

		TEST_METHOD(Propagator_GivenWeightsAndInput_ReturnsOutput)
		{
			double weights[4] = { 1, 2, 3, 4 };
			auto context = GetContext();
			auto& weightMatrix = context->get<Matrix>();
			weightMatrix.Set(2, 2, weights);

			double inputLayer[2] = { 1, 1 };
			auto& inputLayerVector = context->get<Vector>();
			inputLayerVector.Set(2, inputLayer);
			inputLayerVector.SetActivationFunction(PyNet::Models::ActivationFunctionType::Logistic);

			auto& outputLayerVector = context->get<Vector>();
			outputLayerVector.Initialise(2, false);
			outputLayerVector.SetActivationFunction(PyNet::Models::ActivationFunctionType::Logistic);

			double biasesLayer[2] = { 0.1, 0.5 };
			auto& biasesVector = context->get<Vector>();
			biasesVector.Set(2, biasesLayer);

			auto layerPropagator = context->get<LayerPropagator>();
			layerPropagator.PropagateLayer(weightMatrix, inputLayerVector, biasesVector, outputLayerVector);

			Assert::AreEqual(0.95689274505891386, outputLayerVector.GetValue(0));
			Assert::AreEqual(0.99944722136307640, outputLayerVector.GetValue(1));
		}
	};
}