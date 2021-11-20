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
			weightMatrix.Initialise(2, 2);
			weightMatrix = weights;

			double inputLayer[2] = { 1, 1 };
			auto& inputLayerVector = context->get<Vector>();
			inputLayerVector.Initialise(2);
			inputLayerVector.SetActivationFunction(PyNet::Models::ActivationFunctionType::Logistic);

			auto& outputLayerVector = context->get<Vector>();
			outputLayerVector.Initialise(2);
			outputLayerVector.SetActivationFunction(PyNet::Models::ActivationFunctionType::Logistic);

			auto& biasesVector = context->get<Vector>();
			biasesVector.Initialise(2);

			auto layerPropagator = context->get<LayerPropagator>();
			layerPropagator.PropagateLayer(&weightMatrix, &inputLayerVector, &biasesVector, &outputLayerVector);

			Assert::AreEqual(0.95257412682243336, outputLayerVector.GetValue(0));
			Assert::AreEqual(0.99908894880559940, outputLayerVector.GetValue(1));
		}
	};
}