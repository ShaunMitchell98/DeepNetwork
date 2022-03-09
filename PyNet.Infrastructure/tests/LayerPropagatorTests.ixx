module;
#include <CppUnitTest.h>
export module PyNet.Infrastructure.Tests:LayerPropagatorTests;

import PyNet.Infrastructure;
import PyNet.Models;
import :UnitTest;

using namespace PyNet::Models;
using namespace PyNet::Infrastructure;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

TEST_CLASS(LayerPropagatorTests), public UnitTest
{
public:

	TEST_METHOD(Propagator_GivenWeightsAndInput_ReturnsOutput)
	{
		double weights[4] = { 1, 2, 3, 4 };
		auto weightMatrix = GetUniqueService<Matrix>();
		weightMatrix->Set(2, 2, weights);

		double inputLayer[2] = { 1, 1 };
		auto inputLayerVector = GetUniqueService<Vector>();
		inputLayerVector->Set(2, inputLayer);

		auto outputLayerVector = GetUniqueService<Vector>();
		outputLayerVector->Initialise(2, false);

		double biasesLayer[2] = { 0.1, 0.5 };
		auto biasesVector = GetUniqueService<Vector>();
		biasesVector->Set(2, biasesLayer);

		auto layerPropagator = GetUniqueService<LayerPropagator>();
		layerPropagator->PropagateLayer(*weightMatrix, *inputLayerVector, *biasesVector, *outputLayerVector);

		Assert::AreEqual(0.95689274505891386, (*outputLayerVector)[0]);
		Assert::AreEqual(0.99944722136307640, (*outputLayerVector)[1]);
	}
};