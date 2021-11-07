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
			auto weightMatrix = std::make_unique<PyNet::Models::Matrix>(2, 2, weights, true);

			double inputLayer[2] = { 1, 1 };
			auto inputLayerVector = std::make_unique<PyNet::Models::Vector>(2, inputLayer, ActivationFunctions::ActivationFunctionType::Logistic, true);

			auto outputLayerVector = std::make_unique<PyNet::Models::Vector>(2, ActivationFunctions::ActivationFunctionType::Logistic, true);
			auto biasesVector = std::make_unique<PyNet::Models::Vector>(2, true);

			auto context = GetContext(false, false);
			auto layerPropagator = context->get<LayerPropagator>();
			layerPropagator->PropagateLayer(weightMatrix.get(), inputLayerVector.get(), biasesVector.get(), outputLayerVector.get());

			Assert::AreEqual(0.95257412682243336, outputLayerVector->GetValue(0));
			Assert::AreEqual(0.99908894880559940, outputLayerVector->GetValue(1));
		}
	};
}