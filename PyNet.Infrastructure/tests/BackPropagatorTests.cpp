#include "UnitTest.h"
#include <gtest/gtest.h>
#include "BackPropagator.h"
#include "Layers/InputLayer.h"
#include "Layers/FakeLayer.h"

using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure::Tests
{
	class BackPropagatorTests : public UnitTest {};

	TEST_F(BackPropagatorTests, Propagator_GivenNetwork_UpdatesLayers)
	{
		auto backPropagator = GetUniqueService<BackPropagator>();
		auto pyNetwork = GetUniqueService<PyNetwork>();

		auto inputLayer = GetUniqueService<InputLayer>();
		auto fakeLayer = GetUniqueService<FakeLayer>();

		pyNetwork->Layers.push_back(move(inputLayer));
		pyNetwork->Layers.push_back(move(fakeLayer));

		auto lossDerivative = GetUniqueService<Matrix>();
		backPropagator->Propagate(*pyNetwork, *lossDerivative);

		auto finalLayer = static_cast<FakeLayer*>(pyNetwork->Layers.back().get());
		ASSERT_TRUE(finalLayer->Adjusted);
	}
}