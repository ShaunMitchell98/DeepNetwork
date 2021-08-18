#include "CppUnitTest.h"
#include "../DeepNetwork.Infrastructure/Forward Propagation/LayerPropagator.h"
#include "FakeLogger.h"
#include <memory>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DeepNetworkTests
{
	TEST_CLASS(LayerPropagatorTests)
	{
	public:

		TEST_METHOD(Propagator_GivenWeightsAndInput_ReturnsOutput)
		{
			double weights[4] = { 1, 2, 3, 4 };
			auto weightMatrix = std::make_unique<Matrix>(2, 2, weights);

			double inputLayer[2] = { 1, 1 };
			auto inputLayerVector = std::make_unique<Vector>(2, inputLayer, ActivationFunctionType::Logistic);

			auto outputLayerVector = std::make_unique<Vector>(2, ActivationFunctionType::Logistic);

			auto layerPropagator = std::make_unique<LayerPropagator>(std::make_shared<FakeLogger>());
			layerPropagator->PropagateLayer(weightMatrix.get(), inputLayerVector.get(), outputLayerVector.get());

			Assert::AreEqual(0.9525741268, outputLayerVector->GetValue(0));
			Assert::AreEqual(0.9990889488, outputLayerVector->GetValue(1));
		}
	};
}
